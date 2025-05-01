#!/usr/bin/env python
# Custom PPO implementation with InfiniViT for ViZDoom My Way Home
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torchvision.transforms.functional as TF
from collections import deque
from vizdoom import gymnasium_wrapper
from transformers import BertModel, BertConfig

# === PPO Agent ===
class PPOAgent(nn.Module):
    def __init__(self, cnn_encoder, action_dim, hidden_dim=512, context_length=32):
        super().__init__()
        self.cnn = cnn_encoder
        self.feature_buffer = FeatureBuffer(context_length, cnn_encoder.out_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.hidden_dim = hidden_dim
        self.flattened_dim = cnn_encoder.out_dim
        self.context_length = context_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT model
        self.bert_config = BertConfig(
            vocab_size=10000,  # 10000 is the default vocabulary size for BERT
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=self.flattened_dim * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
        )
        self.bert = BertModel(self.bert_config).to(self.device)

        self.policy = nn.Linear(self.flattened_dim, action_dim).to(self.device)
        self.value = nn.Linear(self.flattened_dim, 1).to(self.device)

    def forward(self, obs):
        # Ensure obs is on the correct device
        obs = obs.to(self.device) if isinstance(obs, torch.Tensor) else obs
        # Support both single and batched obs
        if obs.dim() == 4:
            # Batched input: [batch, C, H, W]
            batch_size = obs.shape[0]
            feats = self.cnn(obs)  # [batch, out_dim]
            outputs = []
            for i in range(batch_size):
                feat = feats[i].view(-1)
                self.feature_buffer.append(feat)
                feature_seq = self.feature_buffer.get_sequence()
                feature_seq = torch.stack(feature_seq, dim=0).unsqueeze(0).to(self.device)
                bert_output = self.bert(inputs_embeds=feature_seq)
                bert_hidden_state = bert_output.last_hidden_state
                cls_token = bert_hidden_state[:, 0, :]
                
                logits = self.policy(cls_token)
                value = self.value(cls_token)
                outputs.append((logits, value.squeeze(-1), feat))
            # Stack outputs
            logits = torch.cat([o[0] for o in outputs], dim=0)
            values = torch.cat([o[1] for o in outputs], dim=0)
            feats = torch.stack([o[2] for o in outputs], dim=0)
            return logits, values, feats
        else:
            # Single input
            feat = self.cnn(obs)
            feat = feat.view(-1)
            self.feature_buffer.append(feat)
            feature_seq = self.feature_buffer.get_sequence()
            feature_seq = torch.stack(feature_seq, dim=0).unsqueeze(0).to(self.device)
            bert_output = self.bert(inputs_embeds=feature_seq)
            bert_hidden_state = bert_output.last_hidden_state
            cls_token = bert_hidden_state[:, 0, :]
            logits = self.policy(cls_token)
            value = self.value(cls_token)
            return logits, value.squeeze(-1), feat

    def get_action(self, obs):
        # Ensure we're in evaluation mode for inference
        self.eval()
        with torch.no_grad():
            logits, value, feat = self.forward(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value, feat

# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values, self.features = [], [], [], []
        self.advantages, self.returns = [], []

    def store(self, obs, action, reward, done, log_prob, value, feature):
        # Ensure obs is a tensor, not a dictionary
        if isinstance(obs, dict) and 'screen' in obs:
            obs = torch.tensor(obs['screen'], dtype=torch.float32, device=self.obs[0].device if self.obs else torch.device("cuda" if torch.cuda.is_available() else "cpu")).permute(2, 0, 1)
        # Detach tensors to avoid backprop through the same graph
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(action, torch.Tensor):
            action = action.detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(value, torch.Tensor):
            value = value.detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.features.append(feature)

    def finish_path(self, gamma=0.99, lam=0.95):
        adv, ret = [], []
        gae = 0
        next_value = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            adv.insert(0, gae)
            ret.insert(0, gae + self.values[i])
            next_value = self.values[i]
        self.advantages = adv
        self.returns = ret

    def get_batches(self, batch_size):
        n = len(self.obs)
        indices = np.arange(n)
        np.random.shuffle(indices)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for start in range(0, n, batch_size):
            end = start + batch_size
            yield [
                torch.stack([self.obs[i] for i in indices[start:end]]).to(device),
                torch.tensor([self.actions[i] for i in indices[start:end]]).to(device),
                torch.tensor([self.log_probs[i] for i in indices[start:end]]).to(device),
                torch.tensor([self.advantages[i] for i in indices[start:end]]).to(device),
                torch.tensor([self.returns[i] for i in indices[start:end]]).to(device),
                torch.stack([self.features[i] for i in indices[start:end]]).to(device)
            ]

    def clear(self):
        self.__init__()

class FeatureBuffer:
    # Buffer to store the CNN features for the transformer context window
    def __init__(self, context_length, feature_dim, device):
        self.context_length = context_length
        self.feature_dim = feature_dim
        self.device = device
        self.buffer = deque(maxlen=context_length)

    def append(self, feature):
        # Ensure feature is 1D and correct shape
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32, device=self.device)
        feature = feature.view(-1).to(self.device)
        if feature.shape[0] != self.feature_dim:
            raise ValueError(f"FeatureBuffer: feature has shape {feature.shape}, expected ({self.feature_dim},)")
        self.buffer.append(feature)

    def get_sequence(self):
        # Build sequence with padding and detach older features
        seq = list(self.buffer)
        # Detach all but the most recent feature to avoid reusing old graphs
        for i in range(len(seq) - 1):
            seq[i] = seq[i].detach().to(self.device)
        # Pad with zeros if needed
        if len(seq) < self.context_length:
            pad_size = self.context_length - len(seq)
            pad = [torch.zeros(self.feature_dim, device=self.device)] * pad_size
            seq = pad + seq
        return seq
    
    def clear(self):
        self.buffer.clear()

# === ViT Feature Extractor Wrapper ===
class CNNFeatureWrapper(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flattened_dim = 0
        self.buffer = []
        self.out_dim = out_dim
        # InfiniViT configuration
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        ).to(self.device)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 240, 320, device=self.device)
            test_output = self.cnn(test_input)
            self.flattened_dim = test_output.view(1, -1).shape[1]
        self.projection = nn.Linear(self.flattened_dim, out_dim).to(self.device)
    def forward(self, obs):
        # Handle both dictionary observations and direct arrays
        if isinstance(obs, dict):
            obs = obs['screen']
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            # Convert from HWC to CHW if needed
            if obs_tensor.shape[-1] == 3 and obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.permute(2, 0, 1)
        else:
            obs_tensor = obs.to(self.device)
            if obs_tensor.shape[-1] == 3 and obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.permute(2, 0, 1)
        # Normalize
        obs_tensor = obs_tensor / 255.0
        logits = self.cnn(obs_tensor)
        logits = logits.view(logits.size(0), -1)
        logits = self.projection(logits) # Downsamples to out_dim for memory efficiency
        return logits

# Function to save model checkpoints
def save_checkpoint(path, agent, optimizer, timesteps, rewards):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'cnn_encoder_state_dict': agent.cnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards
    }
    torch.save(checkpoint, path)

# === Training Loop ===
def train(env_id="VizdoomMyWayHome-v0", total_timesteps=500_000, rollout_len=4096, batch_size=64, 
          K_epochs=4, gamma=0.99, lam=0.95, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
          lr=3e-4, max_grad_norm=0.5, save_dir="models", window_size=10):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_cnn_transformer_best.pt")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Initialize models
    cnn_encoder = CNNFeatureWrapper().to(device)
    agent = PPOAgent(cnn_encoder, env.action_space.n).to(device)
    print("Models initialized on device")

    # Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    # Buffer
    buffer = RolloutBuffer()

    # Initial state
    obs, _ = env.reset()
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    episodes_completed = 0
    script_start_time = time.time()
    total_episodes = 0
    
    # Model saving variables
    best_mean_reward = float('-inf')
    
    # Training loop
    for iteration in range(0, total_timesteps, rollout_len):
        iteration_start_time = time.time()
        buffer.clear()
        episode_reward = 0
        episode_length = 0
        local_episodes_completed = 0

        # Set agent to evaluation mode for rollout
        agent.eval()
        
        # Collect experience
        for t in range(rollout_len):
            # Get action
            obs_screen = obs['screen'] if isinstance(obs, dict) else obs
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=device)
            if len(obs_tensor.shape) == 3 and obs_tensor.shape[0] != 3:
                obs_tensor = obs_tensor.permute(2, 0, 1)
                
            with torch.no_grad():
                action, log_prob, _, value, feature = agent.get_action(obs_tensor.unsqueeze(0))

            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1

            # Store transition
            buffer.store(obs_tensor, action.item(), reward, done, log_prob.item(), value.item(), feature.squeeze(0))
            
            obs = next_obs

            # Reset if episode is done
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                local_episodes_completed += 1
                episode_reward = 0
                episode_length = 0
                agent.feature_buffer.clear()

        # Update totals
        total_episodes += local_episodes_completed
        episodes_completed += local_episodes_completed
        
        # Calculate advantages and returns
        buffer.finish_path()
        
        # Set to training mode
        agent.train()
        
        # Collection statistics
        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time
        
        current_timestep = iteration + rollout_len
        
        # Calculate mean reward
        if len(episode_rewards) > 0:
            window = min(window_size, len(episode_rewards))
            mean_reward = np.mean(episode_rewards[-window:])
            mean_length = np.mean(episode_lengths[-window:])
        else:
            mean_reward = float('nan')
            mean_length = float('nan')
            
        # SB3-style logging
        time_elapsed = int(time.time() - script_start_time)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"---------------------------------")
        print(f"| rollout/                |       |")
        print(f"|    ep_len_mean          | {mean_length:<5.1f} |")
        print(f"|    ep_rew_mean          | {mean_reward:<5.3f} |")
        print(f"|    best_mean_reward     | {best_mean_reward:<5.3f} |")
        print(f"|-------------------------|-------|")
        print(f"| time/                   |       |")
        print(f"|    fps                  | {fps:<5.1f} |")
        print(f"|    iteration            | {iteration//rollout_len + 1:<5} |")
        print(f"|    time_elapsed         | {time_elapsed:<5} |")
        print(f"|    total_timesteps      | {current_timestep:<5} |")
        print(f"|-------------------------|-------|")
        
        # Save best model
        if mean_reward > best_mean_reward and len(episode_rewards) >= window_size:
            best_mean_reward = mean_reward
            save_checkpoint(model_path, agent, optimizer, current_timestep, episode_rewards)
            print(f"New best model! Mean reward: {mean_reward:.3f}. Saving to {model_path}")
        
        # Perform multiple epochs of training
        agent.train()
        
        policy_losses, value_losses, entropy_losses = [], [], []
        
        for epoch in range(K_epochs):
            for obs_batch, act_batch, logp_batch, adv_batch, ret_batch, feat_batch in buffer.get_batches(batch_size):
                # Move data to device
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                logp_batch = logp_batch.to(device)
                adv_batch = adv_batch.to(device)
                ret_batch = ret_batch.to(device)
                
                # Normalize advantages
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
                
                # Forward pass
                logits, values, _ = agent(obs_batch)
                
                # Calculate policy loss
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(act_batch)
                
                # Calculate KL divergence and clip fraction (for monitoring)
                ratio = torch.exp(new_log_probs - logp_batch)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean().item()
                
                # PPO objective
                ratio = torch.exp(new_log_probs - logp_batch)
                clip_adv = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
                policy_loss = -torch.min(ratio * adv_batch, clip_adv).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, ret_batch)
                
                # Entropy bonus
                entropy_loss = dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
                
                # Update agent
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
                # Record losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Print training statistics in SB3 style
        print(f"| train/                  |       |")
        print(f"|    approx_kl            | {approx_kl:<5.3f} |")
        print(f"|    clip_fraction        | {clip_fraction:<5.3f} |")
        print(f"|    entropy_loss         | {np.mean(entropy_losses):<5.3f} |")
        print(f"|    learning_rate        | {current_lr:<5.5f} |")
        print(f"|    policy_loss          | {np.mean(policy_losses):<5.3f} |")
        print(f"|    value_loss           | {np.mean(value_losses):<5.3f} |")
        print(f"---------------------------------")
                
    # Close environment
    env.close()
    print("\nTraining completed!")
    
    # Final statistics
    if len(episode_rewards) > 0:
        print(f"\n{'='*30} Final Statistics {'='*30}")
        print(f"Total episodes: {total_episodes}")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Best mean reward: {best_mean_reward:.3f}")
        print(f"Total timesteps: {current_timestep}")
        print(f"Total training time: {time.time() - script_start_time:.1f}s")
        print(f"Best model saved to: {model_path}")
    else:
        print("No episodes completed during training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO InfiniViT")
    parser.add_argument("--env_id", type=str, default="VizdoomMyWayHome-v0")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--rollout_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="models")
    args = parser.parse_args()

    train(env_id=args.env_id, total_timesteps=args.total_timesteps,
          rollout_len=args.rollout_len, batch_size=args.batch_size,
          K_epochs=args.K_epochs, gamma=args.gamma, lam=args.lam,
          clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
          lr=args.lr, max_grad_norm=args.max_grad_norm, save_dir=args.save_dir)