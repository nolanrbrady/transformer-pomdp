import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from vizdoom import gymnasium_wrapper
import types
from models.basic_vit import BasicViT
from models.huggingface_vit_policy import TinyViTPolicy, TinyViTActorCriticPolicy

class REINFORCEAgent:
    """
    REINFORCE agent using a ViT as the policy network.
    """
    def __init__(self, env, policy, lr=1e-4, gamma=0.99):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        
    def collect_trajectory(self, max_steps=1000, render=False):
        """Collect a single episode trajectory."""
        obs, info = self.env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        
        for t in range(max_steps):
            # Extract screen from observation dictionary
            screen = obs['screen']
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1) / 255.0  # (C, H, W)
            
            # Get action from policy - using training=True to maintain gradients
            action, log_prob = self.policy.act(screen_tensor, training=True)
            
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
        
        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps)
            
            # Calculate total episode reward
            total_reward = sum(trajectory['rewards'])
            episode_rewards.append(total_reward)
            
            # Update policy
            loss = self.update_policy(trajectory)
            
            # Print progress
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Loss = {loss:.4f}")

            total_rewards.append(total_reward)
            total_losses.append(loss)
            
            # Early stopping if environment is solved
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
    def __init__(self, env, policy, value_head, lr=1e-4, gamma=0.99):
        self.env = env
        self.policy = policy  # Actor (policy) network
        self.value_head = value_head  # Critic (value) network
        
        # Initialize optimizer based on available networks
        if value_head is None:
            # When using a combined actor-critic policy (like HuggingFaceActorCriticPolicy)
            self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        else:
            # When using separate policy and value_head networks
            self.optimizer = optim.Adam(list(policy.parameters()) + list(value_head.parameters()), lr=lr)
            
        self.gamma = gamma
    
    def get_value(self, state):
        """Get value estimate for a state."""
        with torch.no_grad():
            return self.value_head(state).item()
    
    def collect_trajectory(self, max_steps=1000):
        """Collect a single episode trajectory with value estimates."""
        obs, info = self.env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        for t in range(max_steps):
            # Extract screen from observation dictionary
            screen = obs['screen']
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1) / 255.0
            
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
        mini_batch_size=64
    ):
        self.env = env
        self.policy = policy
        self.value_head = value_head
        
        # Initialize optimizer based on available networks
        if value_head is None:
            # When using a combined actor-critic policy (like HuggingFaceActorCriticPolicy)
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
    
    def collect_trajectories(self, num_steps=2048):
        """Collect multiple trajectories for a total of approximately num_steps transitions."""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        obs, info = self.env.reset()
        
        for t in range(num_steps):
            # Extract screen from observation dictionary
            screen = obs['screen']
            
            # Convert screen to tensor and normalize
            screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1) / 255.0
            
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
                action_logits = self.policy(mb_observations)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                
                # Get log probs of actions from old policy
                new_log_probs = dist.log_prob(mb_actions)
                
                # Calculate entropy for exploration regularization
                entropy = dist.entropy().mean()
                
                # Forward pass through value network
                values = self.value_head(mb_observations).squeeze()
                
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
                    list(self.policy.parameters()) + list(self.value_head.parameters()), 
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
            
            # Normalize if needed
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
    env_id = "VizdoomBasic-v0"
    env = gym.make(env_id, render_mode="human")
    
    # Get observation and action dimensions
    obs_space = env.observation_space['screen']
    act_space = env.action_space.n
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
    
    # Model settings for TinyViT
    model_settings = {
        'img_size': (img_height, img_width),
        'in_channels': channels,
        'num_classes': act_space,
        'resize_inputs': True
    }
    
    print("Using non-pretrained tiny ViT model (faster training)")
    
    if algorithm == "reinforce":
        # Use the TinyViT for REINFORCE (actor only)
        print("\nTraining with REINFORCE algorithm using TinyViT")
        policy = TinyViTPolicy(**model_settings)
        agent = REINFORCEAgent(env=env, policy=policy, lr=0.0001, gamma=0.99)
        rewards = agent.train(num_episodes=1500, max_steps=1000)
        
    elif algorithm == "ac":
        # Use the TinyViT with Actor-Critic head
        print("\nTraining with Actor-Critic algorithm using TinyViT")
        policy = TinyViTActorCriticPolicy(**model_settings)
        # Since our TinyViTActorCriticPolicy includes both actor and critic,
        # we pass it directly to the ActorCriticAgent
        agent = ActorCriticAgent(env=env, policy=policy, value_head=None, lr=0.0001, gamma=0.99)
        
        # Need to monkey-patch the ActorCriticAgent since it expects separate policy and value_head
        # We override the get_value method to use our combined policy
        original_get_value = agent.get_value
        def new_get_value(self, state):
            with torch.no_grad():
                _, value = self.policy.forward(state)
                return value.item()
        agent.get_value = types.MethodType(new_get_value, agent)
        
        rewards = agent.train(num_episodes=1000, max_steps=1000)
        
    elif algorithm == "ppo":
        # Use the TinyViT with Actor-Critic head for PPO
        print("\nTraining with PPO algorithm using TinyViT")
        policy = TinyViTActorCriticPolicy(**model_settings)
        
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
            mini_batch_size=64
        )
        
        # Override evaluate method to use our policy's evaluate method
        def new_evaluate(self, obs, actions):
            return self.policy.evaluate(obs, actions)
        agent.evaluate = types.MethodType(new_evaluate, agent)
        
        rewards = agent.train(num_iterations=1500, steps_per_iteration=2048)
        
    else:
        print("Invalid algorithm choice. Please choose 'reinforce', 'ac', or 'ppo'.")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    train_vizdoom_basic() 