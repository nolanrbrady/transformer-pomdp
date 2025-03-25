import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from vizdoom import gymnasium_wrapper
from models.temporal_vit import TemporalViT


# Helper function to find common divisors for patch size
def find_common_divisors(a, b, min_val=8, max_val=32):
    common = []
    for i in range(min_val, min(max_val + 1, min(a, b) + 1)):
        if a % i == 0 and b % i == 0:
            common.append(i)
    return common


###############################
# Temporal REINFORCE Agent
###############################
class TemporalREINFORCEAgent:
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
        # Initialize observation window using the first frame
        first_image = obs['screen']
        first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]
        
        for t in range(max_steps):
            image = obs['screen']
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            obs_window.pop(0)
            obs_window.append(image_tensor)
            
            # Create a temporal batch of shape (T, C, H, W)
            temporal_batch = torch.cat(obs_window, dim=0)
            
            # Use policy.act which returns (action, log_prob, value); ignore value
            action, log_prob, _ = self.policy.act(temporal_batch)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            if terminated or truncated:
                break
        return {"actions": actions, "log_probs": log_probs, "rewards": rewards}
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def update_policy(self, trajectory):
        log_probs = torch.stack(trajectory['log_probs'])
        returns = self.compute_returns(trajectory['rewards'])
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = - (log_probs * returns).mean()
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


###############################
# Temporal Actor-Critic Agent
###############################
class TemporalActorCriticAgent:
    def __init__(self, env, policy, lr=1e-4, gamma=0.99, temporal_window_size=5):
        self.env = env
        self.policy = policy  # TemporalViT that outputs (action, log_prob, value)
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
            
            # Create a temporal batch
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = - (log_probs * advantages).mean()
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


###############################
# Temporal PPO Agent
###############################
class TemporalPPOAgent:
    def __init__(self, env, policy, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                 gae_lambda=0.95, update_epochs=4, mini_batch_size=64, temporal_window_size=5):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.temporal_window_size = temporal_window_size

    def collect_trajectories(self, num_steps=2048):
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        obs, info = self.env.reset()
        # Initialize observation window
        first_image = obs['screen']
        first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1) / 255.0
        obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]
        
        for _ in range(num_steps):
            image = obs['screen']
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            obs_window.pop(0)
            obs_window.append(image_tensor)
            temporal_batch = torch.cat(obs_window, dim=0)  # (T, C, H, W)
            observations.append(temporal_batch)
            
            action, log_prob, value = self.policy.act(temporal_batch)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, info = self.env.reset()
                first_image = obs['screen']
                first_tensor = torch.FloatTensor(first_image).permute(2, 0, 1) / 255.0
                obs_window = [first_tensor.clone() for _ in range(self.temporal_window_size)]
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones
        }
    
    def compute_advantages_and_returns(self, rewards, values, dones):
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_value * (1 - d) - v.item()
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages.insert(0, gae)
            next_value = v.item()
            ret = gae + v.item()
            returns.insert(0, ret)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def evaluate_policy(self, obs_batch, actions_batch):
        # Evaluate a batch by processing each temporal observation individually
        new_log_probs = []
        values = []
        entropies = []
        for obs, action in zip(obs_batch, actions_batch):
            # obs is of shape (T, C, H, W); process it with forward
            logits, value = self.policy.forward(obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_prob = dist.log_prob(torch.tensor(action))
            new_log_probs.append(new_log_prob)
            values.append(value)
            entropies.append(dist.entropy())
        new_log_probs = torch.stack(new_log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)
        return new_log_probs, values, entropies
    
    def update_policy(self, trajectories):
        obs = trajectories['observations']
        old_actions = trajectories['actions']
        old_log_probs = torch.stack(trajectories['log_probs']).detach()
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        values = trajectories['values']
        
        advantages, returns = self.compute_advantages_and_returns(rewards, values, dones)
        batch_size = len(obs)
        indices = np.arange(batch_size)
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                mb_indices = indices[start:end]
                mb_obs = [obs[i] for i in mb_indices]
                mb_actions = [old_actions[i] for i in mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                new_log_probs, mb_values, entropy = self.evaluate_policy(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(mb_values.squeeze(), mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.mean().item()}
    
    def train(self, total_iterations=50, steps_per_iteration=2048):
        for itr in range(total_iterations):
            trajectories = self.collect_trajectories(steps_per_iteration)
            loss_info = self.update_policy(trajectories)
            mean_reward = np.mean(trajectories['rewards'])
            print(f"Iteration {itr+1}: Mean Reward = {mean_reward:.2f}, Policy Loss = {loss_info['policy_loss']:.4f}, Value Loss = {loss_info['value_loss']:.4f}, Entropy = {loss_info['entropy']:.4f}")


###############################
# Main function to select algorithm and train
###############################

def main():
    env_id = "VizdoomBasic-v0"
    env = gym.make(env_id, render_mode="human")
    
    # Get observation and action dimensions
    obs_space = env.observation_space['screen']
    act_space = env.action_space.n
    img_height, img_width, channels = obs_space.shape
    print(f"Image dimensions: {img_height}x{img_width}x{channels}, Actions: {act_space}")
    
    # Determine appropriate patch size
    divisors = find_common_divisors(img_height, img_width)
    patch_size = max(divisors) if divisors else 16
    print(f"Using patch size: {patch_size}")
    
    algorithm = input("Choose algorithm (reinforce/ac/ppo): ").strip().lower()
    
    if algorithm == "reinforce":
        policy = TemporalViT(
            img_size=(img_height, img_width),
            patch_size=patch_size,
            in_channels=channels,
            num_classes=act_space,
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            pad_if_needed=True,
        )
        agent = TemporalREINFORCEAgent(env, policy, lr=1e-4, gamma=0.99, temporal_window_size=5)
        agent.train(num_episodes=2000, max_steps=1500)
    elif algorithm in ["ac", "ppo"]:
        from actor_critic import ActorCriticVit
        policy = ActorCriticVit(
            img_size=(img_height, img_width),
            patch_size=patch_size,
            in_channels=channels,
            num_classes=act_space,
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            pad_if_needed=True,
        )
        if algorithm == "ac":
            agent = TemporalActorCriticAgent(env, policy, lr=1e-4, gamma=0.99, temporal_window_size=5)
            agent.train(num_episodes=2000, max_steps=1500)
        else:
            agent = TemporalPPOAgent(env, policy, lr=3e-4, gamma=0.99, clip_ratio=0.2, value_coef=0.5,
                                      entropy_coef=0.01, gae_lambda=0.95, update_epochs=4, mini_batch_size=64,
                                      temporal_window_size=5)
            agent.train(total_iterations=2000, steps_per_iteration=2048)
    else:
        print("Invalid algorithm choice!")
    
    env.close()


if __name__ == "__main__":
    main() 