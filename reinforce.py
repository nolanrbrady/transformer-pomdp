import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from vizdoom import gymnasium_wrapper  
from torch.utils.data import TensorDataset, DataLoader
from models.temporal_vit import SimpleViT
########################################
# Policy Network for REINFORCE (Actor-only)
########################################


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        """
        A simple feed-forward network that outputs action logits.
        You can swap this out with any other architecture (e.g., CNN, RNN, Transformer).
        """
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, act_dim)

    def forward(self, x):
        x = self.net(x)
        logits = self.action_head(x)
        return logits

    def act(self, obs):
        """
        Given an observation, sample an action.
        Returns the action and the log probability of that action.
        """
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

########################################
# REINFORCE Agent Class (Policy Only)
########################################

class REINFORCEAgent:
    def __init__(self, env, policy, lr=1e-3, gamma=0.99, temporal_window_size=10):
        """
        Initialize the REINFORCE agent with the given environment and policy.
        
        Args:
            env: Gymnasium environment
            policy: Policy network (e.g., SimpleViT)
            lr: Learning rate
            gamma: Discount factor
            temporal_window_size: Number of recent frames to consider for temporal processing
        """
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.temporal_window_size = temporal_window_size

    def collect_trajectory(self, max_steps=1000):
        """
        Run one episode and collect actions, log probabilities, and rewards.
        Uses a rolling window for temporal processing.
        
        Note: We don't need to store all observations for REINFORCE, just the
        actions, log_probs, and rewards for credit assignment.
        """
        obs, info = self.env.reset()
        actions, log_probs, rewards = [], [], []
        
        # Initialize observation window with copies of the first frame
        # This creates the initial context for temporal processing
        first_image = obs['screen']
        first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]

        for t in range(max_steps):
            # Extract the image from the observation dictionary
            image = obs['screen']
            
            # Convert image to tensor and normalize
            # Change shape from (H, W, C) to (C, H, W) which is PyTorch's expected format
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Update the observation window (remove oldest, add newest)
            obs_window.pop(0)
            obs_window.append(image_tensor)
            
            # Create a temporal batch from the window
            # Stack along first dimension to create shape [temporal_window_size, C, H, W]
            temporal_batch = torch.cat(obs_window, dim=0)
            
            # Pass the temporal batch to the policy
            action, log_prob = self.policy.act(temporal_batch)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            if terminated or truncated:
                break

        return {
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards
        }

    def compute_returns(self, rewards):
        """
        Compute the discounted returns for the episode.
        """
        returns = []
        G = 0
        # Reverse accumulate the discounted rewards.
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # Normalize returns to improve stability (optional).
        returns = torch.FloatTensor(returns)
        return returns

    def update_policy(self, trajectory):
        """
        Update the policy network using the REINFORCE update rule.
        
        The REINFORCE algorithm only needs the log_probs and rewards for credit 
        assignment - we don't need to use the observations again since the
        policy gradient is calculated using log_probs * returns.
        """
        log_probs = trajectory['log_probs']
        rewards = trajectory['rewards']
        returns = self.compute_returns(rewards)

        # Convert list of log probabilities into a tensor.
        log_probs = torch.stack(log_probs)
        
        # Compute loss: -sum(log_prob * return)
        loss = -torch.sum(log_probs * returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes=1000, max_steps=1000):
        """
        Main training loop over multiple episodes.
        """
        for episode in range(num_episodes):
            trajectory = self.collect_trajectory(max_steps)
            loss = self.update_policy(trajectory)
            total_reward = sum(trajectory['rewards'])
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Loss = {loss:.4f}")

########################################
# Example Usage
########################################

if __name__ == "__main__":
    # Define the environment.
    env_id = "VizdoomBasic-v0"  # Change to your desired environment. For VizDoom, use the appropriate id.
    env = gym.make(env_id, render_mode="human")

    # Get observation and action space dimensions.
    obs_space = env.observation_space['screen']
    act_space = env.action_space.n  # Assuming a discrete action space.

    # Extract image dimensions for ViT model
    img_height, img_width, channels = obs_space.shape
    print(f"Original image dimensions: {img_height}x{img_width}x{channels}")

    # Calculate appropriate patch size that divides both dimensions evenly
    # Find common divisors for both dimensions for patches between 8 and 32
    def find_common_divisors(a, b, min_val=8, max_val=32):
        common_divisors = []
        for i in range(min_val, min(max_val + 1, min(a, b) + 1)):
            if a % i == 0 and b % i == 0:
                common_divisors.append(i)
        return common_divisors

    divisors = find_common_divisors(img_height, img_width)
    if divisors:
        patch_size = max(divisors)  # Choose largest common divisor
    else:
        # If no common divisors, choose a value that works with padding
        patch_size = 16  # Default value

    print(f"Using patch size: {patch_size}")

    # Create the policy network using SimpleViT with padding
    policy = SimpleViT(
        img_size=(img_height, img_width),
        patch_size=patch_size,
        in_channels=channels,
        num_classes=act_space,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        pad_if_needed=True,
    )

    # Instantiate the REINFORCE agent with the environment and policy.
    agent = REINFORCEAgent(
        env=env, 
        policy=policy, 
        lr=1e-4, 
        gamma=0.95,
        temporal_window_size=5  # Process 4 recent frames for temporal context
    )

    # Train the agent.
    agent.train(num_episodes=500, max_steps=1000)

    env.close()