import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from vizdoom import gymnasium_wrapper  
from torch.utils.data import TensorDataset, DataLoader
from models.vanilla_vit import SimpleViT

########################################
# Custom Policy Network (Actor-Critic)
########################################

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        """
        A simple feed-forward network that outputs:
          - Action logits (for categorical action distributions)
          - A state-value estimate
        """
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.shared = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, act_dim)
        self.value_head = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        Forward pass returns logits and value estimate.
        """
        x = self.shared(x)
        logits = self.action_head(x)
        value = self.value_head(x)
        return logits, value

    def act(self, obs):
        """
        Given an observation, sample an action and return:
          - The action (as a scalar)
          - The log probability of that action
          - The state value estimate
        """
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, obs, actions):
        """
        Evaluate actions for a batch of observations.
        Returns:
          - Log probabilities for the given actions
          - State value estimates
          - Entropy of the action distribution
        """
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, value, entropy

########################################
# PPO Agent Class
########################################

class PPOAgent:
    def __init__(self, env, policy, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 update_epochs=10, batch_size=64):
        """
        Initialize the PPO agent with hyperparameters and the given environment and policy.
        """
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def collect_trajectories(self, num_steps):
        """
        Collect trajectories (rollouts) from the environment.
        Returns a dictionary with observations, actions, log probabilities,
        rewards, values, and done flags.
        """
        obs_list, actions_list, log_probs_list = [], [], []
        rewards_list, values_list, dones_list = [], [], []
        
        obs, info = self.env.reset()
        for _ in range(num_steps):
            # Assume observations are flattened. (If they are images, you may need to preprocess.)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = self.policy.act(obs_tensor)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob.item())
            rewards_list.append(reward)
            values_list.append(value.item())
            dones_list.append(terminated or truncated)
            
            obs = next_obs
            if terminated or truncated:
                obs, info = self.env.reset()
                
        return {
            'obs': np.array(obs_list),
            'actions': np.array(actions_list),
            'log_probs': np.array(log_probs_list),
            'rewards': np.array(rewards_list),
            'values': np.array(values_list),
            'dones': np.array(dones_list)
        }

    def compute_returns(self, rewards, dones, last_value):
        """
        Compute the discounted returns. Resets the return if a done flag is encountered.
        """
        returns = []
        G = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0  # Reset if the episode ended
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns)

    def update(self, trajectories):
        """
        Update the policy using the collected trajectories.
        Implements the clipped PPO objective.
        """
        obs = torch.FloatTensor(trajectories['obs'])
        actions = torch.LongTensor(trajectories['actions'])
        old_log_probs = torch.FloatTensor(trajectories['log_probs'])
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        values = trajectories['values']
        
        # Get the value of the last observation (used for bootstrapping)
        obs_tensor = torch.FloatTensor(trajectories['obs'][-1]).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.policy.forward(obs_tensor)
        next_value = next_value.item()
        
        returns = self.compute_returns(rewards, dones, next_value)
        returns = torch.FloatTensor(returns)
        advantages = returns - torch.FloatTensor(values)
        
        dataset = TensorDataset(obs, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for _ in range(self.update_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
                new_log_probs, values, entropy = self.policy.evaluate(batch_obs, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, total_steps, update_interval):
        """
        Main training loop. Collect trajectories for update_interval steps,
        then perform policy updates.
        """
        steps = 0
        while steps < total_steps:
            trajectories = self.collect_trajectories(update_interval)
            self.update(trajectories)
            steps += update_interval
            print(f"Total steps: {steps}")

########################################
# Example Usage
########################################

if __name__ == "__main__":
    # Define environment ID (e.g., Vizdoom scenario). This agent is general and can work
    # with any gymnasium environment.
    env_id = "VizdoomCorridor-v0"
    env = gym.make(env_id, render_mode="human")

    # Get observation and action space dimensions.
    # (If using image observations, consider adding preprocessing to flatten or use CNNs.)
    obs_space = env.observation_space
    act_space = env.action_space.n  # assumes discrete action space

    # Create the policy network.
    # Swap out PolicyNetwork with a different architecture if desired.
    height, width, channels = obs_space['screen'].shape
    # policy = PolicyNetwork(obs_dim=height*width*channels, act_dim=act_space, hidden_sizes=(128, 128))
    policy = SimpleViT(
        img_size=height,
        patch_size=width,
        in_channels=channels,
        num_classes=act_space,
        embed_dim=768,
        context_length=128,  # Context window of 128 as requested
        num_heads=12,
        dropout=0.1,
    ) 
    # Instantiate PPOAgent with the environment and policy.
    ppo_agent = PPOAgent(env, policy, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=10, batch_size=64)
    
    # Train for a total number of steps (e.g., 10,000) updating every 2048 steps.
    ppo_agent.train(total_steps=10000, update_interval=2048)
    
    # Inference (generalized, can be used with any gym environment)
    obs, info = env.reset()
    done = False
    while not done:
        obs = obs['screen']
        print("Observation shape: ", obs.shape)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, _, _ = policy.act(obs_tensor)
        obs, reward, terminated, truncated, info = env.step(action)
        # Render the environment if desired (e.g., for VizDoom, use "human" render mode)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()