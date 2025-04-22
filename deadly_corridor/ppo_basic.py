#!/usr/bin/env python
# Custom PPO implementation with BasicViT for ViZDoom Deadly Corridor
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torchvision.transforms.functional as TF
from vizdoom import gymnasium_wrapper

# Import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.basic_vit import BasicViT

# === PPO Agent ===
class PPOAgent(nn.Module):
    def __init__(self, vit_encoder, action_dim, hidden_dim=256):
        super().__init__()
        self.vit = vit_encoder
        self.policy = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        # Get features from ViT encoder
        feat = self.vit(obs)
        # Forward through policy and value networks
        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value.squeeze(-1), feat

    def get_action(self, obs):
        # Ensure we're in evaluation mode for inference
        self.eval()
        with torch.no_grad():
            logits, value, feat = self.forward(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            # Return the value tensor directly
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
            obs = torch.tensor(obs['screen'], dtype=torch.float32).permute(2, 0, 1)

        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        # Clone and detach value before storing
        self.values.append(value.clone().detach())
        self.features.append(feature.clone().detach())  # Make sure to detach features too

    def finish_path(self, gamma=0.99, lam=0.95):
        adv, ret = [], []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(self.rewards))):
            current_value_tensor = self.values[i]
            delta = self.rewards[i] + gamma * next_value * (1.0 - self.dones[i]) - current_value_tensor.item()
            gae = delta + gamma * lam * (1.0 - self.dones[i]) * gae
            current_return = gae + current_value_tensor.item()
            adv.insert(0, torch.tensor(gae, dtype=torch.float32))
            ret.insert(0, torch.tensor(current_return, dtype=torch.float32))
            next_value = current_value_tensor.item()
        self.advantages = adv
        self.returns = ret

    def get_batches(self, batch_size):
        n = len(self.obs)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)  # Ensure we don't go out of bounds
            batch_indices = indices[start:end]
            yield [
                torch.stack([self.obs[i] for i in batch_indices]),
                torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long), 
                torch.tensor([self.log_probs[i] for i in batch_indices], dtype=torch.float32),  # Ensure float32
                torch.stack([self.advantages[i] for i in batch_indices]), 
                torch.stack([self.returns[i] for i in batch_indices]), 
                torch.stack([self.features[i] for i in batch_indices])
            ]

    def clear(self):
        self.__init__()

# === ViT Feature Extractor Wrapper ===
class ViTFeatureWrapper(nn.Module):
    def __init__(self, img_size=(84, 84)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        # BasicViT configuration
        self.vit = BasicViT(
            img_size=self.img_size,
            patch_size=14,
            in_channels=3,
            num_classes=512,
            embed_dim=512,
            num_blocks=6,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1,
            pad_if_needed=True,
            device=self.device
        ).to(self.device)

    def forward(self, obs):
        # Handle both dictionary observations and direct arrays
        if isinstance(obs, dict):
            obs_screen = obs['screen']
        else:
            obs_screen = obs

        # Ensure tensor and correct device
        if isinstance(obs_screen, np.ndarray):
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs_screen.to(self.device)

        # Ensure CHW format
        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] == 3: # HWC -> CHW
            obs_tensor = obs_tensor.permute(2, 0, 1)
        elif obs_tensor.dim() == 4 and obs_tensor.shape[-1] == 3: # BHWC -> BCHW
             obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        elif obs_tensor.dim() == 3 and obs_tensor.shape[0] != 3 : # Maybe grayscale HWC? Error likely
             raise ValueError(f"Unexpected single observation shape: {obs_tensor.shape}")
        elif obs_tensor.dim() == 4 and obs_tensor.shape[1] != 3: # Maybe grayscale BHWC? Error likely
             raise ValueError(f"Unexpected batch observation shape: {obs_tensor.shape}")

        # Normalize and resize
        obs_tensor = obs_tensor / 255.0
        obs_tensor = TF.resize(obs_tensor, list(self.img_size), antialias=True)

        # Add batch dimension if processing a single observation
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)

        # BasicViT expects BCHW
        return self.vit(obs_tensor).squeeze(0) # Remove batch dim if added

# Function to save model checkpoints
def save_checkpoint(path, agent, optimizer, timesteps, rewards):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'vit_encoder_state_dict': agent.vit.state_dict(), # Save wrapped ViT state
        'optimizer_state_dict': optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards
    }
    torch.save(checkpoint, path)

# === Training Loop ===
def train(env_id="VizdoomCorridor-v0", total_timesteps=1_000_000, rollout_len=4096, batch_size=64,
          K_epochs=4, gamma=0.99, lam=0.95, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
          lr=3e-4, max_grad_norm=0.5, save_dir="models", window_size=10):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Adjust save path for Deadly Corridor
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_basic_vit_best.pt")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Initialize models
    vit_encoder = ViTFeatureWrapper().to(device) # Use the wrapper
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
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
            # Prepare observation tensor (handle dict, ensure CHW)
            obs_screen = obs['screen'] if isinstance(obs, dict) else obs
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=device)
            if obs_tensor.dim() == 3 and obs_tensor.shape[-1] == 3: # HWC -> CHW
                obs_tensor = obs_tensor.permute(2, 0, 1)
            
            # Normalize pixel values to [0, 1]
            obs_tensor = obs_tensor / 255.0
            
            # Resize if necessary for the ViT model
            if obs_tensor.shape[1:] != vit_encoder.img_size:
                obs_tensor = TF.resize(obs_tensor, list(vit_encoder.img_size), antialias=True)

            # Get action
            with torch.no_grad():
                # Pass single obs (CHW) to get_action, which handles unsqueezing for batch dim
                action, log_prob, _, value, feature = agent.get_action(obs_tensor)

            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Store transition (pass value tensor, features)
            buffer.store(obs_tensor, action.item(), reward, done, log_prob.item(), value, feature)

            obs = next_obs

            # Reset if episode is done
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                local_episodes_completed += 1
                episode_reward = 0
                episode_length = 0

        # Update totals
        total_episodes += local_episodes_completed
        episodes_completed += local_episodes_completed

        # Calculate advantages and returns
        buffer.finish_path(gamma=gamma, lam=lam)

        # Set to training mode
        agent.train()

        # Collection statistics
        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time if collection_time > 0 else float('inf')

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
        if best_mean_reward != float('-inf'):
             print(f"|    best_mean_reward     | {best_mean_reward:<5.3f} |")
        print(f"|-------------------------|-------|")
        print(f"| time/                   |       |")
        print(f"|    fps                  | {fps:<5.1f} |")
        print(f"|    iteration            | {iteration//rollout_len + 1:<5} |")
        print(f"|    time_elapsed         | {time_elapsed:<5} |")
        print(f"|    total_timesteps      | {current_timestep:<5} |")
        print(f"|-------------------------|-------|")

        # Save best model
        if not np.isnan(mean_reward) and mean_reward > best_mean_reward and len(episode_rewards) >= window_size:
            best_mean_reward = mean_reward
            save_checkpoint(model_path, agent, optimizer, current_timestep, episode_rewards)
            print(f"New best model! Mean reward: {mean_reward:.3f}. Saving to {model_path}")

        # Perform multiple epochs of training
        policy_losses, value_losses, entropy_losses, approx_kls, clip_fractions = [], [], [], [], []

        for epoch in range(K_epochs):
            for obs_batch, act_batch, logp_batch, adv_batch, ret_batch, feat_batch in buffer.get_batches(batch_size):
                # Move data to device
                obs_batch = obs_batch.to(device) # Should be BCHW
                act_batch = act_batch.to(device)
                logp_batch = logp_batch.to(device)
                adv_batch = adv_batch.to(device) # Tensor
                ret_batch = ret_batch.to(device) # Tensor
                
                # Normalize advantages
                if len(adv_batch) > 1:  # Only normalize if batch size > 1
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                # Forward pass - can use features or recompute
                logits, values, _ = agent(obs_batch) # values should be [B]

                # Calculate policy loss
                probs = F.softmax(logits, dim=-1)  # Use F.softmax for numerical stability
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(act_batch)

                # Calculate KL divergence and clip fraction (for monitoring)
                ratio = torch.exp(new_log_probs - logp_batch)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()

                # PPO objective
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
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())

        # Calculate explained variance
        y_pred = np.array([v.cpu().numpy() for v in buffer.values])
        y_true = np.array([r.cpu().numpy() for r in buffer.returns])
        y_var = np.var(y_true)
        explained_var = np.nan if y_var == 0 else 1 - np.var(y_true - y_pred) / y_var


        # Print training statistics in SB3 style (end of epochs)
        print(f"| train/                  |       |")
        print(f"|    approx_kl            | {np.mean(approx_kls):<5.3f} |")
        print(f"|    clip_fraction        | {np.mean(clip_fractions):<5.3f} |")
        print(f"|    entropy_loss         | {np.mean(entropy_losses):<5.3f} |")
        print(f"|    explained_variance   | {explained_var:<5.3f} |")
        print(f"|    learning_rate        | {current_lr:<5.5f} |")
        print(f"|    policy_loss          | {np.mean(policy_losses):<5.3f} |")
        print(f"|    value_loss           | {np.mean(value_losses):<5.3f} |")
        print(f"---------------------------------")

    # Close environment
    env.close()
    print("Training completed!")

    # Final statistics
    if len(episode_rewards) > 0:
        print(f"{'='*30} Final Statistics {'='*30}")
        print(f"Total episodes: {total_episodes}")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Best mean reward: {best_mean_reward:.3f}")
        print(f"Total timesteps: current_timestep")
        print(f"Total training time: {time.time() - script_start_time:.1f}s")
        print(f"Best model saved to: {model_path}")
    else:
        print("No episodes completed during training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO BasicViT for Deadly Corridor")
    # Adjusted default env_id
    parser.add_argument("--env_id", type=str, default="VizdoomCorridor-v0")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
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
    parser.add_argument("--window_size", type=int, default=10) # Add window size arg
    args = parser.parse_args()

    train(env_id=args.env_id, total_timesteps=args.total_timesteps,
          rollout_len=args.rollout_len, batch_size=args.batch_size,
          K_epochs=args.K_epochs, gamma=args.gamma, lam=args.lam,
          clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
          lr=args.lr, max_grad_norm=args.max_grad_norm, save_dir=args.save_dir,
          window_size=args.window_size) # Pass window size




