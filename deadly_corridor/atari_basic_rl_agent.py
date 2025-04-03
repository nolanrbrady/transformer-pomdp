import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from vizdoom import gymnasium_wrapper
import types
from models.basic_vit import BasicViT
import ale_py

gym.register_envs(ale_py)

# Only need to extend BasicViT for actor-critic functionality
class BasicViTWithValue(BasicViT):
    """
    Extends BasicViT to include a value head for Actor-Critic methods.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add a value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize the value head weights
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def get_value(self, x):
        """
        Get state value estimates.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns: values of shape (B, 1)
        """
        # Make sure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32)
            
        # Handle single image case by adding batch dimension
        if len(x.shape) == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
        
        # Normalize if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # Get patch embeddings and run through transformer blocks
        features = self.patch_embed(x)
        for block in self.blocks:
            features = block(features)
        features = self.norm(features)
        
        # Use CLS token for value prediction
        features = features[:, 0]  # Get [CLS] token: (B, embed_dim)
        
        # Value head
        values = self.value_head(features)  # (B, 1)
        
        return values.squeeze(-1)
    
    def act_with_value(self, x, training=False):
        """
        Sample an action and return log probability and value estimate.
        For Actor-Critic methods.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        training: If True, gradient computation is enabled
        """
        if not training:
            with torch.no_grad():
                # Get action and log prob from the standard act method
                action, log_prob = self.act(x)
                # Get value separately
                value = self.get_value(x)
                return action, log_prob, value
        else:
            # During training, we need to keep gradients
            # Normalize if not already done (handled in self.forward)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.get_value(x)
            return action.item(), log_prob, value
    
    def evaluate(self, x, actions):
        """
        Evaluate actions for given states, returning log probabilities, values, and entropy.
        For PPO and Actor-Critic methods.
        x: Tensor of shape (B, C, H, W)
        actions: Tensor of shape (B,) containing action indices
        """
        # Get logits and values
        logits = self.forward(x)
        values = self.get_value(x)
        
        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Compute log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy

class REINFORCEAgent:
    """
    REINFORCE agent using a ViT as the policy network.
    """
    def __init__(self, env, policy, lr=1e-4, gamma=0.99, is_vector_env=False, num_envs=1):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.is_vector_env = is_vector_env
        self.num_envs = num_envs
        
    def collect_trajectory(self, max_steps=1000, render=False):
        """Collect a single episode trajectory."""
        if self.is_vector_env:
            return self.collect_trajectory_parallel(max_steps)
            
        # Original single-env implementation
        obs, info = self.env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        
        for t in range(max_steps):
            # Extract screen from observation dictionary
            screen = obs
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1)  # (C, H, W)
            
            # Get action from policy - using training=True to maintain gradients
            action, log_prob = self.policy.act(screen_tensor)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            states.append(screen_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            # Update observation
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones
        }
    
    def collect_trajectory_parallel(self, max_steps=1000):
        """Collect multiple trajectories in parallel using vectorized environment."""
        # Reset all environments
        observations, infos = self.env.reset()
        
        # Initialize lists for each environment
        all_states = [[] for _ in range(self.num_envs)]
        all_actions = [[] for _ in range(self.num_envs)]
        all_log_probs = [[] for _ in range(self.num_envs)]
        all_rewards = [[] for _ in range(self.num_envs)]
        all_dones = [[] for _ in range(self.num_envs)]
        
        # Keep track of which environments are done
        env_done = [False] * self.num_envs
        total_done = 0
        
        for t in range(max_steps):
            # Convert observations to tensor
            batch_observations = torch.FloatTensor(observations).permute(0, 3, 1, 2)  # (num_envs, C, H, W)
            
            # Get actions for all environments
            batch_actions = []
            batch_log_probs = []
            
            # Process each environment individually to maintain gradients properly
            for i in range(self.num_envs):
                if env_done[i]:
                    # Skip environments that are done
                    batch_actions.append(0)  # Dummy action
                    batch_log_probs.append(torch.zeros(1))
                    continue
                
                action, log_prob = self.policy.act(batch_observations[i], training=True)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
            
            # Convert to numpy for environment step
            numpy_actions = np.array(batch_actions)
            
            # Step all environments
            next_observations, rewards, terminated, truncated, infos = self.env.step(numpy_actions)
            dones = np.logical_or(terminated, truncated)
            
            # Store transitions for each environment
            for i in range(self.num_envs):
                if env_done[i]:
                    continue
                    
                all_states[i].append(batch_observations[i])
                all_actions[i].append(batch_actions[i])
                all_log_probs[i].append(batch_log_probs[i])
                all_rewards[i].append(rewards[i])
                all_dones[i].append(dones[i])
                
                # Check if environment is now done
                if dones[i] and not env_done[i]:
                    env_done[i] = True
                    total_done += 1
            
            # Update observations
            observations = next_observations
            
            # Break if all environments are done
            if total_done == self.num_envs:
                break
        
        # Combine all complete episodes into a single batch
        combined_states = []
        combined_actions = []
        combined_log_probs = []
        combined_rewards = []
        combined_dones = []
        
        for i in range(self.num_envs):
            if len(all_rewards[i]) > 0:  # Only add if this environment collected some experience
                combined_states.extend(all_states[i])
                combined_actions.extend(all_actions[i])
                combined_log_probs.extend(all_log_probs[i])
                combined_rewards.extend(all_rewards[i])
                combined_dones.extend(all_dones[i])
        
        return {
            'states': combined_states,
            'actions': combined_actions,
            'log_probs': combined_log_probs,
            'rewards': combined_rewards,
            'dones': combined_dones
        }
    
    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def update_policy(self, trajectory):
        """Update policy using REINFORCE (policy gradient)."""
        log_probs = torch.stack(trajectory['log_probs'])
        returns = self.compute_returns(trajectory['rewards'])
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        policy_loss = -(log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train(self, num_episodes=500, max_steps=1000):
        """Train the agent for a specified number of episodes."""
        episode_rewards = []
        total_rewards = []
        total_losses = []
        
        # Calculate actual episodes needed for vector environments
        episodes_per_update = 1 if not self.is_vector_env else self.num_envs
        
        for episode in range(0, num_episodes, episodes_per_update):
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps)
            
            # Calculate total episode reward
            if self.is_vector_env:
                # With parallel environments, we have multiple episodes per update
                episode_end_indices = [i for i, done in enumerate(trajectory['dones']) if done]
                if not episode_end_indices:
                    episode_end_indices = [len(trajectory['rewards'])]
                
                # Calculate rewards per episode
                start_idx = 0
                episode_total_rewards = []
                
                for end_idx in episode_end_indices:
                    if end_idx > start_idx:  # Ensure we have at least one step
                        episode_total_rewards.append(sum(trajectory['rewards'][start_idx:end_idx]))
                        start_idx = end_idx
                
                # Average reward across completed episodes
                total_reward = sum(episode_total_rewards) / len(episode_total_rewards) if episode_total_rewards else 0
            else:
                total_reward = sum(trajectory['rewards'])
            
            episode_rewards.append(total_reward)
            
            # Update policy
            loss = self.update_policy(trajectory)
            
            # Print progress
            print(f"Episode {episode+1}-{episode+episodes_per_update}: Reward = {total_reward:.2f}, Loss = {loss:.4f}")
            
            total_rewards.append(total_reward)
            total_losses.append(loss)
            
            # Early stopping if environment is solved (for CartPole)
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195:
                print(f"Environment solved in {episode+1} episodes!")
                break
        
        # Save the total rewards and losses to a file
        total_rewards_array = np.array(total_rewards)
        total_losses_array = np.array(total_losses)
        np.save('basic_reinforce_total_rewards.npy', total_rewards_array)
        np.save('basic_reinforce_total_losses.npy', total_losses_array)
        
        return episode_rewards

class ActorCriticAgent:
    """
    Actor-Critic agent using BasicViT with an additional value head.
    """
    def __init__(self, env, policy, value_head, lr=1e-4, gamma=0.99, is_vector_env=False, num_envs=1):
        self.env = env
        self.policy = policy  # Actor (policy) network
        self.value_head = value_head  # Critic (value) network
        
        # Initialize optimizer based on available networks
        if value_head is None:
            # When using a combined actor-critic policy (like BasicViTWithValue)
            self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        else:
            # When using separate policy and value_head networks
            self.optimizer = optim.Adam(list(policy.parameters()) + list(value_head.parameters()), lr=lr)
            
        self.gamma = gamma
        self.is_vector_env = is_vector_env
        self.num_envs = num_envs
    
    def get_value(self, state):
        """Get value estimate for a state."""
        with torch.no_grad():
            return self.value_head(state).item()
    
    def collect_trajectory(self, max_steps=1000):
        """Collect a single episode trajectory with value estimates."""
        if self.is_vector_env:
            return self.collect_trajectory_parallel(max_steps)
        
        # Original single-env code
        obs, info = self.env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        for t in range(max_steps):
            # Extract screen from observation dictionary
            screen = obs
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1)
            
            # Get action from policy and value estimate
            if self.value_head is None:
                # Use the combined actor-critic policy
                action, log_prob, value = self.policy.act(screen_tensor, training=True)
            else:
                # Use separate policy and value networks
                action, log_prob = self.policy.act(screen_tensor, training=True)
                value = self.value_head(screen_tensor.unsqueeze(0))
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            states.append(screen_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(terminated or truncated)
            
            # Update observation
            obs = next_obs
            
            if terminated or truncated:
                break
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones
        }
    
    def collect_trajectory_parallel(self, max_steps=1000):
        """Collect trajectories from parallel environments."""
        # Reset all environments
        observations, infos = self.env.reset()
        
        # Initialize lists to store trajectories for each environment
        all_states = [[] for _ in range(self.num_envs)]
        all_actions = [[] for _ in range(self.num_envs)]
        all_log_probs = [[] for _ in range(self.num_envs)]
        all_rewards = [[] for _ in range(self.num_envs)]
        all_values = [[] for _ in range(self.num_envs)]
        all_dones = [[] for _ in range(self.num_envs)]
        
        # Track which environments are done
        env_done = [False] * self.num_envs
        total_done = 0
        
        for t in range(max_steps):
            # Process batch of observations
            batch_observations = torch.FloatTensor(observations).permute(0, 3, 1, 2)  # (num_envs, C, H, W)
            
            # Forward pass through policy to get actions for all environments
            batch_actions = []
            batch_log_probs = []
            batch_values = []
            
            # Process each environment separately to maintain gradients properly
            for i in range(self.num_envs):
                if env_done[i]:
                    # Skip environments that are already done
                    batch_actions.append(0)  # Dummy action
                    batch_log_probs.append(torch.zeros(1))
                    batch_values.append(torch.zeros(1))
                    continue
                    
                action, log_prob, value = self.policy.act(batch_observations[i], training=True)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)
            
            # Convert to numpy array for environment step
            numpy_actions = np.array(batch_actions)
            
            # Step all environments
            next_observations, rewards, terminated, truncated, infos = self.env.step(numpy_actions)
            dones = np.logical_or(terminated, truncated)
            
            # Store transitions for each environment
            for i in range(self.num_envs):
                if env_done[i]:
                    continue
                    
                all_states[i].append(batch_observations[i])
                all_actions[i].append(batch_actions[i])
                all_log_probs[i].append(batch_log_probs[i])
                all_rewards[i].append(rewards[i])
                all_values[i].append(batch_values[i])
                all_dones[i].append(dones[i])
                
                # Mark environment as done if episode terminated
                if dones[i] and not env_done[i]:
                    env_done[i] = True
                    total_done += 1
            
            # Update observations
            observations = next_observations
            
            # Break if all environments are done
            if total_done == self.num_envs:
                break
        
        # Combine all trajectories into a single batch
        combined_states = []
        combined_actions = []
        combined_log_probs = []
        combined_rewards = []
        combined_values = []
        combined_dones = []
        
        # Only use complete episodes
        for i in range(self.num_envs):
            if len(all_rewards[i]) > 0:  # Only add if environment collected some experience
                combined_states.extend(all_states[i])
                combined_actions.extend(all_actions[i])
                combined_log_probs.extend(all_log_probs[i])
                combined_rewards.extend(all_rewards[i])
                combined_values.extend(all_values[i])
                combined_dones.extend(all_dones[i])
        
        return {
            'states': combined_states,
            'actions': combined_actions,
            'log_probs': combined_log_probs,
            'rewards': combined_rewards,
            'values': combined_values,
            'dones': combined_dones
        }
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """Compute returns and advantages."""
        returns = []
        advantages = []
        next_value = 0  # Terminal state has value 0
        next_return = 0
        
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            # Compute return (discounted sum of rewards)
            next_return = r + self.gamma * next_return * (1 - done)
            returns.insert(0, next_return)
            
            # Compute advantage (return - value)
            next_advantage = next_return - v.item()
            advantages.insert(0, next_advantage)
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
    
    def update_policy(self, trajectory):
        """Update policy and value function using Actor-Critic."""
        states = trajectory['states']
        log_probs = torch.stack(trajectory['log_probs'])
        values = torch.cat(trajectory['values'])
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute actor (policy) loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Compute critic (value) loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Combined loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Update policy and value function
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=500, max_steps=1000):
        """Train the agent for a specified number of episodes."""
        episode_rewards = []
        
        total_rewards = []
        total_losses = []
        
        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps)
            
            # Calculate total episode reward
            total_reward = sum(trajectory['rewards'])
            episode_rewards.append(total_reward)
            
            # Update policy and value function
            loss = self.update_policy(trajectory)
            
            # Print progress
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Loss = {loss:.4f}")
            
            total_rewards.append(total_reward)
            total_losses.append(loss)

        # Save the total rewards and losses to a file
        total_rewards_array = np.array(total_rewards)
        total_losses_array = np.array(total_losses)
        np.save('basic_ac_total_rewards.npy', total_rewards_array)
        np.save('basic_ac_total_losses.npy', total_losses_array)

        return episode_rewards

class PPOAgent:
    """
    Proximal Policy Optimization agent using BasicViT with an additional value head.
    """
    def __init__(
        self, 
        env, 
        policy, 
        value_head, 
        lr=3e-4, 
        gamma=0.99, 
        clip_ratio=0.2, 
        value_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        update_epochs=4,
        mini_batch_size=64,
        is_vector_env=False,
        num_envs=1
    ):
        self.env = env
        self.policy = policy
        self.value_head = value_head
        
        # Initialize optimizer based on available networks
        if value_head is None:
            # When using a combined actor-critic policy (like BasicViTWithValue)
            self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        else:
            # When using separate policy and value_head networks
            self.optimizer = optim.Adam(list(policy.parameters()) + list(value_head.parameters()), lr=lr)
        
        # PPO parameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        # Parallel environment parameters
        self.is_vector_env = is_vector_env
        self.num_envs = num_envs
    
    def collect_trajectories(self, num_steps=2048):
        """Collect multiple trajectories for a total of approximately num_steps transitions."""
        if self.is_vector_env:
            return self.collect_trajectories_parallel(num_steps)
        
        # Original single-env code
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        obs, info = self.env.reset()
        
        for t in range(num_steps):
            # Extract screen from observation dictionary
            screen = obs
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1)
            
            # Get action from policy
            with torch.no_grad():  # Use no_grad here since we're just collecting trajectories
                if self.value_head is None:
                    # Combined actor-critic policy
                    action, log_prob, value = self.policy.act(screen_tensor)
                else:
                    # Separate policy and value networks
                    action, log_prob = self.policy.act(screen_tensor)
                    value = self.value_head(screen_tensor.unsqueeze(0)).squeeze()
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            observations.append(screen_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(terminated or truncated)
            
            # Update observation
            obs = next_obs
            
            if terminated or truncated:
                obs, info = self.env.reset()
        
        # Calculate advantages and returns
        advantages, returns = self.compute_advantages_and_returns(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones,
            'advantages': advantages,
            'returns': returns
        }
    
    def collect_trajectories_parallel(self, num_steps=2048):
        """Collect trajectories from parallel environments."""
        # Storage for collected data
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        # Reset all environments
        obs, info = self.env.reset()
        
        # Calculate steps per environment
        steps_per_env = max(1, num_steps // self.num_envs)
        
        for t in range(steps_per_env):
            # Process batch of observations
            batch_observations = torch.FloatTensor(obs).permute(0, 3, 1, 2)  # (num_envs, C, H, W)
            
            # Get actions for all environments in one batch
            with torch.no_grad():
                batch_actions = []
                batch_log_probs = []
                batch_values = []
                
                for i in range(self.num_envs):
                    action, log_prob, value = self.policy.act(batch_observations[i])
                    batch_actions.append(action)
                    batch_log_probs.append(log_prob)
                    batch_values.append(value)
            
            # Convert to numpy array for environment step
            numpy_actions = np.array(batch_actions)
            
            # Step all environments
            next_obs, batch_rewards, terminated, truncated, infos = self.env.step(numpy_actions)
            batch_dones = np.logical_or(terminated, truncated)
            
            # Store transitions from all environments
            for i in range(self.num_envs):
                observations.append(batch_observations[i])
                actions.append(batch_actions[i])
                log_probs.append(batch_log_probs[i])
                rewards.append(batch_rewards[i])
                values.append(batch_values[i])
                dones.append(batch_dones[i])
            
            # Update observations
            obs = next_obs
            
            # No need to reset environments, vectorized env handles this automatically
        
        # Calculate advantages and returns
        advantages, returns = self.compute_advantages_and_returns(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones,
            'advantages': advantages,
            'returns': returns
        }
    
    def compute_advantages_and_returns(self, rewards, values, dones, gamma, gae_lambda):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = []
        returns = []
        gae = 0
        
        # Convert to numpy for faster computation
        rewards = np.array(rewards)
        values = torch.stack(values).detach().cpu().numpy()
        dones = np.array(dones)
        
        # Add a dummy final value (zero) since we don't have the value of the next state
        # for the last state in our batch
        values = np.append(values, 0)
        
        for t in reversed(range(len(rewards))):
            # Delta = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            # GAE_t = delta_t + gamma * lambda * (1 - done_t) * GAE_{t+1}
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            
            # Insert at the beginning since we're iterating in reverse
            advantages.insert(0, gae)
            
            # Return = GAE + value
            returns.insert(0, gae + values[t])
        
        # Convert back to tensors
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, memory):
        """
        Update policy and value function using PPO clipped objective.
        This is the core of the PPO algorithm and performs multiple updates on the same batch.
        """
        observations = memory['observations']
        old_actions = memory['actions']
        old_log_probs = torch.stack(memory['log_probs']).detach()
        advantages = memory['advantages']
        returns = memory['returns']
        
        batch_size = len(observations)
        indices = np.arange(batch_size)
        
        # Perform multiple optimization epochs
        for _ in range(self.update_epochs):
            # Shuffle the data
            np.random.shuffle(indices)
            
            # Process the data in mini-batches
            for start_idx in range(0, batch_size, self.mini_batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Create mini-batch tensors
                mb_observations = torch.stack([observations[i] for i in mb_indices])
                mb_actions = torch.LongTensor([old_actions[i] for i in mb_indices])
                mb_old_log_probs = torch.stack([old_log_probs[i] for i in mb_indices])
                mb_advantages = torch.stack([advantages[i] for i in mb_indices])
                mb_returns = torch.stack([returns[i] for i in mb_indices])
                
                # Forward pass through policy network to get new action distributions
                if self.value_head is None:
                    # For TinyViTActorCriticPolicy, which returns logits from forward
                    action_logits = self.policy(mb_observations)
                    # Get values from the policy's get_value method
                    values = self.policy.get_value(mb_observations)
                else:
                    # For separate policy and value networks
                    action_logits = self.policy(mb_observations)
                    values = self.value_head(mb_observations).squeeze()
                
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                
                # Get log probs of actions from old policy
                new_log_probs = dist.log_prob(mb_actions)
                
                # Calculate entropy for exploration regularization
                entropy = dist.entropy().mean()
                
                # Calculate the ratio of new and old policies
                # r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Calculate policy loss (negative because we're maximizing)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy and value networks
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: clip gradients for stability
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    max_norm=0.5
                )
                self.optimizer.step()
        
        # Return the final loss values
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(self, num_iterations=1500, steps_per_iteration=2048):
        """Train the agent for a specified number of iterations."""
        total_steps = 0
        episode_rewards = []
        episode_reward = 0
        
        obs, info = self.env.reset()

        total_rewards = []
        total_losses = []
        
        for iteration in range(num_iterations):
            # Collect trajectories
            memory = self.collect_trajectories(steps_per_iteration)
            
            # Calculate metrics
            mean_reward = np.mean(memory['rewards'])
            total_steps += len(memory['rewards'])
            
            # Update policy and value function
            losses = self.update_policy(memory)
            
            # Print progress
            print(f"Iteration {iteration+1}: ")
            print(f"  Total steps: {total_steps}")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Policy loss: {losses['policy_loss']:.4f}")
            print(f"  Value loss: {losses['value_loss']:.4f}")
            print(f"  Entropy: {losses['entropy']:.4f}")

            total_rewards.append(mean_reward)
            total_losses.append(losses['policy_loss'] + losses['value_loss'] + losses['entropy'])

        # Save the total rewards and losses to a file
        total_rewards_array = np.array(total_rewards)
        total_losses_array = np.array(total_losses)
        np.save('basic_ppo_total_rewards.npy', total_rewards_array)
        np.save('basic_ppo_total_losses.npy', total_losses_array)
            
        return episode_rewards

def create_value_head(vit_model):
    """Create a value network using the same encoder as the policy network."""
    class ValueHead(nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            # Share the ViT backbone but not the head
            self.patch_embed = vit_model.patch_embed
            self.blocks = vit_model.blocks
            self.norm = vit_model.norm
            # New value head
            self.value_head = nn.Linear(vit_model.embed_dim, 1)
            # Initialize the value head
            nn.init.xavier_uniform_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        
        def forward(self, x):
            # Convert to tensor if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(np.array(x), dtype=torch.float32)
            
            # Handle single image case
            if len(x.shape) == 3:  # (C, H, W)
                x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
            
            # Ensure values are in [0, 1] range
            if x.max() > 1.0:
                x = x / 255.0
            
            # Use the same ViT backbone
            x = self.patch_embed(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            
            # Use CLS token for value prediction
            x = x[:, 0]  # (B, embed_dim)
            
            # Value head
            value = self.value_head(x)  # (B, 1)
            
            return value
    
    return ValueHead(vit_model)

def train_vizdoom_basic():
    """Train agents on VizdoomBasic environment using different algorithms."""
    # Set up environment
    # env_id = "VizdoomBasic-v0"
    env_id = "ALE/Pong-v5"
    
    # Ask user for number of parallel environments
    try:
        num_envs = int(input("Enter number of parallel environments (1 for single env): "))
        num_envs = max(1, num_envs)  # Ensure at least 1 environment
    except ValueError:
        print("Invalid input, using 1 environment")
        num_envs = 1
    
    # Create single environment for observation space inspection
    single_env = gym.make(env_id)
    obs_space = single_env.observation_space
    act_space = single_env.action_space.n
    single_env.close()
    
    # Function to create environment
    def make_env(render_idx=None):
        def _init():
            # Only render the first environment if multiple environments
            render_mode = "human" if render_idx == 0 and num_envs > 0 else None
            return gym.make(env_id, render_mode=render_mode)
        return _init
    
    # Create vectorized environment
    if num_envs > 1:
        print(f"Using {num_envs} parallel environments with AsyncVectorEnv")
        env = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
        is_vector_env = True
    else:
        print("Using single environment")
        env = gym.make(env_id, render_mode="human")
        is_vector_env = False
    
    print("Observation space: ", obs_space)
    print("Action space: ", act_space)
    img_height, img_width, channels = obs_space.shape
    print(f"Image dimensions: {img_height}x{img_width}x{channels}, Actions: {act_space}")
    
    # Function to find common divisors for patch size
    def find_common_divisors(a, b, min_val=8, max_val=32):
        common_divisors = []
        for i in range(min_val, min(max_val + 1, min(a, b) + 1)):
            if a % i == 0 and b % i == 0:
                common_divisors.append(i)
        return common_divisors
    
    # Find appropriate patch size
    divisors = find_common_divisors(img_height, img_width)
    patch_size = max(divisors) if divisors else 16
    print(f"Using patch size: {patch_size}")
    
    # Choose which agent to train
    algorithm = input("Choose algorithm (reinforce/ac/ppo): ").lower()
    
    # Model settings for BasicViT
    model_settings = {
        'img_size': (img_height, img_width),
        'patch_size': patch_size,
        'in_channels': channels,
        'num_classes': act_space,
        'embed_dim': 128,  # Smaller embed dimension for efficiency
        'depth': 1,        # Fewer transformer blocks
        'num_heads': 4,    # Fewer attention heads
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'pad_if_needed': True
    }
    
    print("Using non-pretrained BasicViT model")
    
    if algorithm == "reinforce":
        # Use BasicViT directly for REINFORCE
        print("\nTraining with REINFORCE algorithm using BasicViT")
        policy = BasicViT(**model_settings)
        agent = REINFORCEAgent(env=env, policy=policy, lr=0.0001, gamma=0.99, is_vector_env=is_vector_env, num_envs=num_envs)
        rewards = agent.train(num_episodes=1500, max_steps=1000)
        
    elif algorithm == "ac":
        # Use BasicViTWithValue for Actor-Critic
        print("\nTraining with Actor-Critic algorithm using BasicViT with value head")
        policy = BasicViTWithValue(**model_settings)
        
        # Create ActorCritic agent with our policy
        agent = ActorCriticAgent(env=env, policy=policy, value_head=None, lr=0.0001, gamma=0.99, is_vector_env=is_vector_env, num_envs=num_envs)
        
        # Need to monkey-patch the ActorCriticAgent to use our policy's methods
        original_get_value = agent.get_value
        def new_get_value(self, state):
            with torch.no_grad():
                return self.policy.get_value(state).item()
        agent.get_value = types.MethodType(new_get_value, agent)
        
        # Override the collect_trajectory method to use act_with_value
        original_collect = agent.collect_trajectory
        def new_collect_trajectory(self, max_steps=1000):
            if self.is_vector_env:
                return self.collect_trajectory_parallel(max_steps)
            
            # Original single-env code
            obs, info = self.env.reset()
            
            states = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            dones = []
            
            for t in range(max_steps):
                # Extract screen from observation
                screen = obs
                
                # Convert screen to tensor
                screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1)
                
                # Get action, log_prob, and value
                action, log_prob, value = self.policy.act_with_value(screen_tensor, training=True)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Store transition
                states.append(screen_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(terminated or truncated)
                
                # Update observation
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            return {
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards,
                'values': values,
                'dones': dones
            }
        
        def collect_trajectory_parallel(self, max_steps=1000):
            """Collect trajectories from parallel environments."""
            # Reset all environments
            observations, infos = self.env.reset()
            
            # Initialize lists to store trajectories for each environment
            all_states = [[] for _ in range(self.num_envs)]
            all_actions = [[] for _ in range(self.num_envs)]
            all_log_probs = [[] for _ in range(self.num_envs)]
            all_rewards = [[] for _ in range(self.num_envs)]
            all_values = [[] for _ in range(self.num_envs)]
            all_dones = [[] for _ in range(self.num_envs)]
            
            # Track which environments are done
            env_done = [False] * self.num_envs
            total_done = 0
            
            for t in range(max_steps):
                # Process batch of observations
                batch_observations = torch.FloatTensor(observations).permute(0, 3, 1, 2)  # (num_envs, C, H, W)
                
                # Forward pass through policy to get actions for all environments
                batch_actions = []
                batch_log_probs = []
                batch_values = []
                
                # Process each environment separately to maintain gradients properly
                for i in range(self.num_envs):
                    if env_done[i]:
                        # Skip environments that are already done
                        batch_actions.append(0)  # Dummy action
                        batch_log_probs.append(torch.zeros(1))
                        batch_values.append(torch.zeros(1))
                        continue
                        
                    action, log_prob, value = self.policy.act(batch_observations[i], training=True)
                    batch_actions.append(action)
                    batch_log_probs.append(log_prob)
                    batch_values.append(value)
                
                # Convert to numpy array for environment step
                numpy_actions = np.array(batch_actions)
                
                # Step all environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(numpy_actions)
                dones = np.logical_or(terminated, truncated)
                
                # Store transitions for each environment
                for i in range(self.num_envs):
                    if env_done[i]:
                        continue
                        
                    all_states[i].append(batch_observations[i])
                    all_actions[i].append(batch_actions[i])
                    all_log_probs[i].append(batch_log_probs[i])
                    all_rewards[i].append(rewards[i])
                    all_values[i].append(batch_values[i])
                    all_dones[i].append(dones[i])
                    
                    # Mark environment as done if episode terminated
                    if dones[i] and not env_done[i]:
                        env_done[i] = True
                        total_done += 1
                
                # Update observations
                observations = next_observations
                
                # Break if all environments are done
                if total_done == self.num_envs:
                    break
            
            # Combine all trajectories into a single batch
            combined_states = []
            combined_actions = []
            combined_log_probs = []
            combined_rewards = []
            combined_values = []
            combined_dones = []
            
            # Only use complete episodes
            for i in range(self.num_envs):
                if len(all_rewards[i]) > 0:  # Only add if environment collected some experience
                    combined_states.extend(all_states[i])
                    combined_actions.extend(all_actions[i])
                    combined_log_probs.extend(all_log_probs[i])
                    combined_rewards.extend(all_rewards[i])
                    combined_values.extend(all_values[i])
                    combined_dones.extend(all_dones[i])
            
            return {
                'states': combined_states,
                'actions': combined_actions,
                'log_probs': combined_log_probs,
                'rewards': combined_rewards,
                'values': combined_values,
                'dones': combined_dones
            }
        
        agent.collect_trajectory = types.MethodType(new_collect_trajectory, agent)
        agent.collect_trajectory_parallel = types.MethodType(collect_trajectory_parallel, agent)
        
        rewards = agent.train(num_episodes=1000, max_steps=1000)
        
    elif algorithm == "ppo":
        # Use BasicViTWithValue for PPO
        print("\nTraining with PPO algorithm using BasicViT with value head")
        policy = BasicViTWithValue(**model_settings)
        
        # Create PPO agent with our policy
        agent = PPOAgent(
            env=env, 
            policy=policy, 
            value_head=None,  # Not used with our combined policy 
            lr=3e-4,
            gamma=0.99,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            gae_lambda=0.95,
            update_epochs=4,
            mini_batch_size=64,
            is_vector_env=is_vector_env,
            num_envs=num_envs
        )
        
        # Override evaluate method to use our policy's evaluate method
        def new_evaluate(self, obs, actions):
            return self.policy.evaluate(obs, actions)
        agent.evaluate = types.MethodType(new_evaluate, agent)
        
        # Override collect_trajectories to handle both single and parallel environments
        original_collect = agent.collect_trajectories
        def new_collect_trajectories(self, num_steps=2048):
            if self.is_vector_env:
                return self.collect_trajectories_parallel(num_steps)
            
            # Original single-env code
            observations = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            dones = []
            
            obs, info = self.env.reset()
            
            for t in range(num_steps):
                # Extract screen from observation dictionary
                screen = obs
                
                # Convert screen to tensor and normalize
                screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1)
                
                # Get action from policy
                with torch.no_grad():  # Use no_grad here since we're just collecting trajectories
                    action, log_prob, value = self.policy.act_with_value(screen_tensor)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Store transition
                observations.append(screen_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(terminated or truncated)
                
                # Update observation
                obs = next_obs
                
                if terminated or truncated:
                    obs, info = self.env.reset()
            
            # Calculate advantages and returns
            advantages, returns = self.compute_advantages_and_returns(
                rewards, values, dones, self.gamma, self.gae_lambda
            )
            
            return {
                'observations': observations,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards,
                'values': values,
                'dones': dones,
                'advantages': advantages,
                'returns': returns
            }
        
        def collect_trajectories_parallel(self, num_steps=2048):
            """Collect trajectories from parallel environments."""
            # Storage for collected data
            observations = []
            actions = []
            log_probs = []
            rewards = []
            values = []
            dones = []
            
            # Reset all environments
            obs, info = self.env.reset()
            
            # Calculate steps per environment
            steps_per_env = max(1, num_steps // self.num_envs)
            
            for t in range(steps_per_env):
                # Process batch of observations
                batch_observations = torch.FloatTensor(obs).permute(0, 3, 1, 2)  # (num_envs, C, H, W)
                
                # Get actions for all environments in one batch
                with torch.no_grad():
                    batch_actions = []
                    batch_log_probs = []
                    batch_values = []
                    
                    for i in range(self.num_envs):
                        action, log_prob, value = self.policy.act_with_value(batch_observations[i])
                        batch_actions.append(action)
                        batch_log_probs.append(log_prob)
                        batch_values.append(value)
                
                # Convert to numpy array for environment step
                numpy_actions = np.array(batch_actions)
                
                # Step all environments
                next_obs, batch_rewards, terminated, truncated, infos = self.env.step(numpy_actions)
                batch_dones = np.logical_or(terminated, truncated)
                
                # Store transitions from all environments
                for i in range(self.num_envs):
                    observations.append(batch_observations[i])
                    actions.append(batch_actions[i])
                    log_probs.append(batch_log_probs[i])
                    rewards.append(batch_rewards[i])
                    values.append(batch_values[i])
                    dones.append(batch_dones[i])
                
                # Update observations
                obs = next_obs
                
                # No need to reset environments, vectorized env handles this automatically
            
            # Calculate advantages and returns
            advantages, returns = self.compute_advantages_and_returns(
                rewards, values, dones, self.gamma, self.gae_lambda
            )
            
            return {
                'observations': observations,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards,
                'values': values,
                'dones': dones,
                'advantages': advantages,
                'returns': returns
            }
        
        agent.collect_trajectories = types.MethodType(new_collect_trajectories, agent)
        agent.collect_trajectories_parallel = types.MethodType(collect_trajectories_parallel, agent)
        
        rewards = agent.train(num_iterations=1500, steps_per_iteration=2048)
        
    else:
        print("Invalid algorithm choice. Please choose 'reinforce', 'ac', or 'ppo'.")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    train_vizdoom_basic() 