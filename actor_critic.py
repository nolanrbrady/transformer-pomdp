import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from vizdoom import gymnasium_wrapper  

from models.temporal_vit import TemporalViT


class ActorCriticVit(TemporalViT):
    """
    Actor-Critic Vision Transformer that extends SimpleViT to output both action logits and a value estimate.
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
        Given a temporal batch of images, sample an action and return the log probability and value estimate.
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


class ActorCriticAgent:
    """
    A basic Actor-Critic agent that uses the ActorCriticVit network as its policy.
    """
    def __init__(self, env, policy, lr=1e-4, gamma=0.99, temporal_window_size=5):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.temporal_window_size = temporal_window_size

    def collect_trajectory(self, max_steps=1000):
        obs, info = self.env.reset()
        actions = []
        log_probs = []
        rewards = []
        values = []
        
        # Initialize observation window using the first frame
        first_image = obs['screen']
        first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]
        
        for t in range(max_steps):
            image = obs['screen']
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            obs_window.pop(0)
            obs_window.append(image_tensor)
            
            # Create a temporal batch of shape (temporal_window_size, C, H, W)
            temporal_batch = torch.cat(obs_window, dim=0)
            
            action, log_prob, value = self.policy.act(temporal_batch)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            
            obs = next_obs
            if terminated or truncated:
                break
            
        return {"actions": actions, "log_probs": log_probs, "rewards": rewards, "values": values}
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def update_policy(self, trajectory, critic_loss_coef=0.5):
        returns = self.compute_returns(trajectory['rewards'])
        log_probs = torch.stack(trajectory['log_probs'])
        values = torch.stack(trajectory['values']).squeeze()
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Consider normalizing advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss: encourage actions with higher advantage
        actor_loss = - (log_probs * advantages).mean()
        
        # Critic loss: minimize mean squared error
        critic_loss = F.mse_loss(values, returns)
        
        loss = actor_loss + critic_loss_coef * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=500, max_steps=1000):
        for episode in range(num_episodes):
            trajectory = self.collect_trajectory(max_steps)
            loss = self.update_policy(trajectory)
            total_reward = sum(trajectory['rewards'])
            print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Loss = {loss:.4f}")


if __name__ == "__main__":
    # Setup Vizdoom Basic environment
    env_id = "VizdoomBasic-v0"
    env = gym.make(env_id, render_mode="human")
    
    # Get observation and action dimensions
    obs_space = env.observation_space['screen']
    act_space = env.action_space.n
    img_height, img_width, channels = obs_space.shape
    print(f"Original image dimensions: {img_height}x{img_width}x{channels}")
    
    # Function to compute common divisors for patch size
    def find_common_divisors(a, b, min_val=8, max_val=32):
        common_divisors = []
        for i in range(min_val, min(max_val + 1, min(a, b) + 1)):
            if a % i == 0 and b % i == 0:
                common_divisors.append(i)
        return common_divisors

    divisors = find_common_divisors(img_height, img_width)
    patch_size = max(divisors) if divisors else 16
    print(f"Using patch size: {patch_size}")
    
    # Create the actor-critic policy network using ActorCriticVit
    policy = ActorCriticVit(
        img_size=(img_height, img_width),
        patch_size=patch_size,
        in_channels=channels,
        num_classes=act_space,  # number of actions
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        pad_if_needed=True,
    )
    
    # Initialize the ActorCriticAgent
    agent = ActorCriticAgent(env=env, policy=policy, lr=0.00005, gamma=0.99, temporal_window_size=1)
    
    # Train the agent
    agent.train(num_episodes=2000, max_steps=1250)
    
    env.close() 