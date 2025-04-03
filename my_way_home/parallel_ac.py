import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from vizdoom import gymnasium_wrapper
import time
from tqdm import tqdm
import os

from models.temporal_vit import TemporalViT

# Set the sharing strategy to file_system for better performance with large tensors
torch.multiprocessing.set_sharing_strategy('file_system')

class ActorCriticVit(TemporalViT):
    """
    Actor-Critic Vision Transformer that extends TemporalViT to output both action logits and a value estimate.
    """
    def __init__(self, *args, **kwargs):
        # Expect kwargs to include num_classes (number of actions)
        super().__init__(*args, **kwargs)
        # Use the original head as the actor head
        self.actor_head = self.head
        # Create a new critic head to predict state value
        self.critic_head = nn.Linear(self.embed_dim, 1)
        # Add initial weights for the action head that don't strongly favor any particular action
        nn.init.uniform_(self.actor_head.weight, -0.03, 0.03)

    def forward(self, x):
        """
        Forward pass for a temporal sequence of frames.
        x: Tensor of shape (T, C, H, W) where T is the temporal window size.
        Returns:
            logits: action logits of shape (1, num_actions)
            value: state value estimate of shape (1,)
        """
        T = x.shape[0]
        spatial_embeddings = []
        for t in range(T):
            img = x[t]  # (C, H, W)
            # Process each image to get a (1, n_patches+1, embed_dim) representation
            embedded_img = self.process_single_image(img)
            # Extract the CLS token representation
            cls_token = embedded_img[:, 0:1, :]
            spatial_embeddings.append(cls_token)
        
        # Stack CLS tokens along temporal dimension: (1, T, embed_dim)
        temporal_sequence = torch.cat(spatial_embeddings, dim=1)

        # Add temporal positional embeddings
        temporal_pos_embed = self.temporal_pos_embed[:, :T, :]
        temporal_sequence = temporal_sequence + temporal_pos_embed
        
        # Process with temporal transformer block
        temporal_sequence = self.temporal_block(temporal_sequence) 
        temporal_sequence = self.temporal_norm(temporal_sequence)
        
        # Use the last token as the final hidden representation
        final_hidden = temporal_sequence[:, -1]  # (1, embed_dim)
        
        # Actor: compute action logits
        logits = self.actor_head(final_hidden)  # (1, num_actions)
        
        # Critic: compute value estimate
        value = self.critic_head(final_hidden).squeeze(-1)  # (1) -> scalar
        return logits, value
    
    def act(self, obs):
        """
        Given an observation, output action, log_prob, and value.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(np.array(obs))
        if len(obs.shape) > 4:
            obs = obs.squeeze(0)
        logits, value = self.forward(obs)  # logits: (1, num_actions), value: (1)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

def make_env(env_id, rank, seed=0, render_mode=None):
    """
    Helper function to create an environment with a given seed
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    return _init

def collect_worker(worker_id, env_fn, policy, device, num_steps, results_queue, temporal_window_size):
    """
    Worker function to collect trajectories from a single environment
    """
    # Create environment
    env = env_fn()
    obs, info = env.reset()
    
    # Initialize observation window using the first frame
    first_image = obs['screen']
    first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1) / 255.0
    obs_window = [first_tensor.clone() for _ in range(temporal_window_size)]
    
    # Data collection
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    
    for step in range(num_steps):
        # Extract screen from observation dictionary
        screen = obs['screen']
        screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1) / 255.0
        
        # Update observation window
        obs_window.pop(0)
        obs_window.append(screen_tensor)
        
        # Create a temporal batch
        temporal_batch = torch.cat(obs_window, dim=0).to(device)
        
        # Get action, log_prob, and value from policy
        with torch.no_grad():
            action, log_prob, value = policy.act(temporal_batch)
        
        # Execute action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        states.append(screen_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(terminated or truncated)
        
        # Update observation
        obs = next_obs
        
        # Reset if episode is done
        if terminated or truncated:
            obs, info = env.reset()
            # Reset observation window
            first_image = obs['screen']
            first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1) / 255.0
            obs_window = [first_tensor.clone() for _ in range(temporal_window_size)]
    
    # Put results in queue
    results_queue.put({
        'worker_id': worker_id,
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'rewards': rewards,
        'values': values,
        'dones': dones
    })
    
    env.close()

class ParallelActorCriticAgent:
    """
    Actor-Critic agent with parallel environment sampling
    """
    def __init__(self, env_id, policy, num_envs=4, lr=1e-4, gamma=0.99, temporal_window_size=4, device='cpu'):
        self.env_id = env_id
        self.policy = policy.to(device)
        self.num_envs = num_envs
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.temporal_window_size = temporal_window_size
        self.device = device
        
        # For rendering during evaluation
        self.eval_env = gym.make(env_id, render_mode="human")
    
    def collect_trajectories_parallel(self, num_steps_per_worker=250):
        """
        Collect trajectories from multiple environments in parallel
        """
        # Create a queue to get results from worker processes
        mp.set_start_method('spawn', force=True)
        results_queue = mp.Queue()
        
        # Create and start worker processes
        processes = []
        for worker_id in range(self.num_envs):
            env_fn = make_env(self.env_id, worker_id)
            p = mp.Process(
                target=collect_worker,
                args=(worker_id, env_fn, self.policy, self.device, num_steps_per_worker, results_queue, self.temporal_window_size)
            )
            p.start()
            processes.append(p)
        
        # Collect results from all workers
        results = []
        for _ in range(self.num_envs):
            results.append(results_queue.get())
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Combine data from all workers
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_dones = []
        
        for result in results:
            all_states.extend(result['states'])
            all_actions.extend(result['actions'])
            all_log_probs.extend(result['log_probs'])
            all_rewards.extend(result['rewards'])
            all_values.extend(result['values'])
            all_dones.extend(result['dones'])
        
        return {
            'states': all_states,
            'actions': all_actions,
            'log_probs': all_log_probs,
            'rewards': all_rewards,
            'values': all_values,
            'dones': all_dones
        }
    
    def compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = []
        R = 0
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
            
        return torch.FloatTensor(returns).to(self.device)
    
    def update_policy(self, trajectory, critic_loss_coef=0.5, entropy_coef=0.01):
        """Update policy using the collected trajectories."""
        # Move data to device (if not already there)
        log_probs = torch.stack([lp.to(self.device) if not lp.device == self.device else lp 
                                for lp in trajectory['log_probs']])
        values = torch.stack([v.to(self.device) if not v.device == self.device else v 
                             for v in trajectory['values']]).squeeze()
        
        # Compute returns and advantages
        returns = self.compute_returns(trajectory['rewards'], trajectory['dones'])
        advantages = returns - values.detach()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss: encourage actions with higher advantage
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss: minimize mean squared error
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy loss to encourage exploration
        logits = self.policy(torch.stack(trajectory['states']).to(self.device))[0]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Combined loss
        loss = actor_loss + critic_loss_coef * critic_loss - entropy_coef * entropy
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }
    
    def evaluate(self, num_episodes=5, max_steps=1000):
        """Evaluate the current policy."""
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, info = self.eval_env.reset()
            episode_reward = 0
            
            # Initialize observation window using the first frame
            first_image = obs['screen']
            first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1) / 255.0
            obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]
            
            for step in range(max_steps):
                # Extract screen
                screen = obs['screen']
                screen_tensor = torch.FloatTensor(screen).permute(2, 0, 1) / 255.0
                
                # Update observation window
                obs_window.pop(0)
                obs_window.append(screen_tensor)
                
                # Create temporal batch
                temporal_batch = torch.cat(obs_window, dim=0).to(self.device)
                
                # Get action
                with torch.no_grad():
                    action, _, _ = self.policy.act(temporal_batch)
                
                # Execute action
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def train(self, total_timesteps=1000000, eval_freq=50000, log_freq=10000, num_steps_per_update=1000):
        """Train the agent."""
        timesteps_per_update = self.num_envs * num_steps_per_update
        num_updates = total_timesteps // timesteps_per_update
        
        all_rewards = []
        timesteps_so_far = 0
        
        start_time = time.time()
        
        for update in range(num_updates):
            # Collect trajectories
            trajectories = self.collect_trajectories_parallel(num_steps_per_worker=num_steps_per_update)
            
            # Update policy
            loss_info = self.update_policy(trajectories)
            
            # Update timesteps
            timesteps_in_update = len(trajectories['rewards'])
            timesteps_so_far += timesteps_in_update
            
            # Calculate mean episode reward
            mean_reward = np.mean(trajectories['rewards'])
            all_rewards.append(mean_reward)
            
            # Log progress
            if update % (log_freq // timesteps_per_update) == 0:
                elapsed_time = time.time() - start_time
                fps = int(timesteps_so_far / elapsed_time)
                
                print(f"Update {update}/{num_updates}, Timesteps: {timesteps_so_far}/{total_timesteps}")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"FPS: {fps}")
                print(f"Loss: {loss_info['loss']:.4f}, Actor Loss: {loss_info['actor_loss']:.4f}, "
                      f"Critic Loss: {loss_info['critic_loss']:.4f}, Entropy: {loss_info['entropy']:.4f}")
                print("-" * 50)
            
            # Evaluate policy
            if update % (eval_freq // timesteps_per_update) == 0:
                eval_reward = self.evaluate(num_episodes=3)
                print(f"Evaluation after {timesteps_so_far} timesteps: Mean Reward = {eval_reward:.2f}")
        
        return all_rewards

def find_common_divisors(a, b, min_val=8, max_val=32):
    """Find common divisors for patch size."""
    common_divisors = []
    for i in range(min_val, min(max_val + 1, min(a, b) + 1)):
        if a % i == 0 and b % i == 0:
            common_divisors.append(i)
    return common_divisors

if __name__ == "__main__":
    # Setup environment parameters
    env_id = "VizdoomBasic-v0"
    test_env = gym.make(env_id)
    
    # Get observation and action dimensions
    obs_space = test_env.observation_space['screen']
    act_space = test_env.action_space.n
    img_height, img_width, channels = obs_space.shape
    print(f"Original image dimensions: {img_height}x{img_width}x{channels}")
    
    # Find appropriate patch size
    divisors = find_common_divisors(img_height, img_width)
    patch_size = max(divisors) if divisors else 16
    print(f"Using patch size: {patch_size}")
    
    # Close test environment
    test_env.close()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create policy network
    policy = ActorCriticVit(
        img_size=(img_height, img_width),
        patch_size=patch_size,
        in_channels=channels,
        num_classes=act_space,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        pad_if_needed=True
    )
    
    # Define number of parallel environments
    num_envs = 4  # Adjust based on your CPU cores
    
    # Create and train the parallel agent
    agent = ParallelActorCriticAgent(
        env_id=env_id,
        policy=policy,
        num_envs=num_envs,
        lr=0.0001,
        gamma=0.99,
        temporal_window_size=4,
        device=device
    )
    
    # Train the agent
    rewards = agent.train(
        total_timesteps=500000,
        eval_freq=50000,
        log_freq=10000,
        num_steps_per_update=250
    )
    
    # Save the trained model
    torch.save(policy.state_dict(), "trained_ac_policy.pt")
    
    print("Training completed and model saved!") 