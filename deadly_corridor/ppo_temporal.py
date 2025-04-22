#!/usr/bin/env python
# Custom PPO implementation with TemporalViT for ViZDoom Deadly Corridor
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

# Import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.temporal_vit import TemporalViT

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
        # Add batch dimension if needed - TemporalViT expects [B, T, C, H, W]
        if obs.dim() == 4:  # [T, C, H, W]
            obs = obs.unsqueeze(0)  # Add batch dimension -> [1, T, C, H, W]
        
        # Forward through TemporalViT
        feat = self.vit.vit(obs)
        
        # Forward through policy and value networks
        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value.squeeze(-1), feat # feat is the feature from the *last* frame in sequence

    def get_action(self, obs_sequence):
        # obs_sequence should be [T, C, H, W]
        self.eval()
        with torch.no_grad():
            logits, value, feat = self.forward(obs_sequence)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            # Return value tensor directly
            return action, dist.log_prob(action), dist.entropy(), value, feat

# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        # Store individual frames, features are calculated later during training
        self.obs_frames, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []
        self.advantages, self.returns = [], []

    def store(self, obs_frame, action, reward, done, log_prob, value):
        # Store the single frame observation tensor
        self.obs_frames.append(obs_frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        # Clone and detach value before storing
        self.values.append(value.clone().detach())

    def finish_path(self, gamma=0.99, lam=0.95):
        adv, ret = [], []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(self.rewards))):
            current_value_tensor = self.values[i]
            delta = self.rewards[i] + gamma * next_value * (1.0 - self.dones[i]) - current_value_tensor
            gae = delta + gamma * lam * (1.0 - self.dones[i]) * gae
            current_return_tensor = gae + current_value_tensor
            adv.insert(0, gae)
            ret.insert(0, current_return_tensor)
            next_value = current_value_tensor.item()
        self.advantages = adv
        self.returns = ret

    def get_batches(self, batch_size, vit_encoder, frame_history):
        n = len(self.obs_frames)
        indices = np.arange(n)
        np.random.shuffle(indices)

        # Need to handle sequence creation for batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            # Prepare sequences for the batch
            batch_sequences = []
            for idx in batch_indices:
                # Find the start index for the sequence (handle boundary conditions)
                seq_start = max(0, idx - frame_history + 1)
                sequence = self.obs_frames[seq_start : idx + 1]
                # Pad if sequence is shorter than frame_history (beginning of rollout)
                if len(sequence) < frame_history:
                    padding_needed = frame_history - len(sequence)
                    # Assume padding with the first frame
                    padding = [sequence[0]] * padding_needed
                    sequence = padding + sequence
                batch_sequences.append(torch.stack(sequence)) # [T, C, H, W]

            # Stack sequences into a batch [B, T, C, H, W]
            obs_batch = torch.stack(batch_sequences)

            # Get features from the ViT encoder for the batch of sequences
            # Note: This recomputes features, potentially slow. Consider storing features if memory allows.
            # REMOVED: Feature calculation moved to agent forward pass
            # with torch.no_grad():
            #     # TemporalViT expects [B, T, C, H, W]
            #     feat_batch = vit_encoder.vit(obs_batch) # [B, Features]

            # Yield batch data
            yield (
                obs_batch, # [B, T, C, H, W]
                torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long),
                torch.tensor([self.log_probs[i] for i in batch_indices]),
                torch.stack([self.advantages[i] for i in batch_indices]),
                torch.stack([self.returns[i] for i in batch_indices]),
                # REMOVED: feat_batch
            )

    def clear(self):
        self.__init__()

# === ViT Feature Extractor Wrapper (Handles Frame Buffering) ===
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history=32, img_size=(84, 84)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.frame_history = frame_history

        # TemporalViT configuration
        self.vit = TemporalViT(
            img_size=self.img_size,
            patch_size=14,
            in_channels=3,
            num_classes=512,
            embed_dim=512,
            num_heads=4,
            mlp_ratio=2.0,
            pad_if_needed=True,
            device=self.device,
            num_spatial_blocks=3,
            num_temporal_blocks=3,
        ).to(self.device)

        # Initialize buffer with empty frames
        self.reset_buffer()

    def reset_buffer(self):
        """Initialize the buffer with zero frames"""
        empty_frame = torch.zeros((3, self.img_size[0], self.img_size[1]),
                                  dtype=torch.float32, device=self.device)
        self.buffer = deque([empty_frame.clone() for _ in range(self.frame_history)],
                            maxlen=self.frame_history)

    def process_observation(self, obs):
         # Handle dict, convert to tensor, normalize, resize, ensure CHW
        if isinstance(obs, dict):
            obs_screen = obs['screen']
        else:
            obs_screen = obs

        if isinstance(obs_screen, np.ndarray):
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs_screen.to(self.device)

        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] == 3: # HWC -> CHW
            obs_tensor = obs_tensor.permute(2, 0, 1)

        obs_tensor = obs_tensor / 255.0
        obs_tensor = TF.resize(obs_tensor, list(self.img_size), antialias=True)
        return obs_tensor # [C, H, W]

    def forward(self, obs_frame):
        """Processes a single observation frame, updates buffer, returns sequence."""
        # obs_frame should be preprocessed: [C, H, W]
        self.buffer.append(obs_frame.clone())
        # Stack frames into a sequence [T, C, H, W]
        stacked = torch.stack(list(self.buffer), dim=0)
        return stacked

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
          lr=3e-4, max_grad_norm=0.5, save_dir="models", window_size=10, frame_history=32):

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_temporal_vit_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    vit_encoder = ViTFeatureWrapper(frame_history=frame_history).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
    print("Models initialized on device")

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    vit_encoder.reset_buffer() # Ensure buffer is reset at start
    current_obs_frame = vit_encoder.process_observation(obs)

    episode_rewards = []
    episode_lengths = []
    episodes_completed = 0
    script_start_time = time.time()
    total_episodes = 0
    best_mean_reward = float('-inf')

    for iteration in range(0, total_timesteps, rollout_len):
        iteration_start_time = time.time()
        buffer.clear()
        episode_reward = 0
        episode_length = 0
        local_episodes_completed = 0

        agent.eval()

        for t in range(rollout_len):
            # Process current observation frame and get sequence
            obs_sequence = vit_encoder.forward(current_obs_frame) # [T, C, H, W]

            with torch.no_grad():
                action, log_prob, _, value, _ = agent.get_action(obs_sequence)

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            # Store the *current* frame (before update) and other info
            buffer.store(current_obs_frame, action.item(), reward, done, log_prob.item(), value)

            episode_reward += reward
            episode_length += 1

            # Prepare next frame
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                vit_encoder.reset_buffer() # Reset buffer on episode end
                current_obs_frame = vit_encoder.process_observation(obs)
                local_episodes_completed += 1
                episode_reward = 0
                episode_length = 0
            else:
                current_obs_frame = vit_encoder.process_observation(next_obs)
                obs = next_obs # Keep original obs for next loop if needed

        total_episodes += local_episodes_completed
        episodes_completed += local_episodes_completed

        buffer.finish_path(gamma=gamma, lam=lam)

        agent.train()

        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time if collection_time > 0 else float('inf')
        current_timestep = iteration + rollout_len

        if len(episode_rewards) > 0:
            window = min(window_size, len(episode_rewards))
            mean_reward = np.mean(episode_rewards[-window:])
            mean_length = np.mean(episode_lengths[-window:])
        else:
            mean_reward = float('nan')
            mean_length = float('nan')

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

        if not np.isnan(mean_reward) and mean_reward > best_mean_reward and len(episode_rewards) >= window_size:
            best_mean_reward = mean_reward
            save_checkpoint(model_path, agent, optimizer, current_timestep, episode_rewards)
            print(f"New best model! Mean reward: {mean_reward:.3f}. Saving to {model_path}")

        policy_losses, value_losses, entropy_losses, approx_kls, clip_fractions = [], [], [], [], []

        for epoch in range(K_epochs):
            # Pass vit_encoder and frame_history to get_batches
            for obs_batch_seq, act_batch, logp_batch, adv_batch, ret_batch in buffer.get_batches(batch_size, vit_encoder, frame_history):
                obs_batch_seq = obs_batch_seq.to(device) # [B, T, C, H, W]
                act_batch = act_batch.to(device)
                logp_batch = logp_batch.to(device)
                adv_batch = adv_batch.to(device)
                ret_batch = ret_batch.to(device)
                # REMOVED: feat_batch = feat_batch.to(device)

                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                # Forward pass using the *features* extracted during get_batches
                # Need PPOAgent.forward to accept features directly OR recompute
                # Let's modify PPOAgent to accept precomputed features if available
                # For now, assume recomputation (simpler but slower):
                logits, values, _ = agent(obs_batch_seq) # Recompute features inside agent.forward

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(act_batch)

                ratio = torch.exp(new_log_probs - logp_batch)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()

                clip_adv = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
                policy_loss = -torch.min(ratio * adv_batch, clip_adv).mean()

                value_loss = F.mse_loss(values, ret_batch)
                entropy_loss = dist.entropy().mean()
                total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())

        # Calculate explained variance based on buffer values/returns
        y_pred = np.array([v.cpu().numpy() for v in buffer.values])
        y_true = np.array([r.cpu().numpy() for r in buffer.returns])
        y_var = np.var(y_true)
        explained_var = np.nan if y_var == 0 else 1 - np.var(y_true - y_pred) / y_var

        print(f"| train/                  |       |")
        print(f"|    approx_kl            | {np.mean(approx_kls):<5.3f} |")
        print(f"|    clip_fraction        | {np.mean(clip_fractions):<5.3f} |")
        print(f"|    entropy_loss         | {np.mean(entropy_losses):<5.3f} |")
        print(f"|    explained_variance   | {explained_var:<5.3f} |")
        print(f"|    learning_rate        | {current_lr:<5.5f} |")
        print(f"|    policy_loss          | {np.mean(policy_losses):<5.3f} |")
        print(f"|    value_loss           | {np.mean(value_losses):<5.3f} |")
        print(f"---------------------------------")

    env.close()
    print("Training completed!")

    if len(episode_rewards) > 0:
        print(f"{'='*30} Final Statistics {'='*30}")
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
    parser = argparse.ArgumentParser(description="PPO TemporalViT for Deadly Corridor")
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
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--frame_history", type=int, default=32)
    args = parser.parse_args()

    train(env_id=args.env_id, total_timesteps=args.total_timesteps,
          rollout_len=args.rollout_len, batch_size=args.batch_size,
          K_epochs=args.K_epochs, gamma=args.gamma, lam=args.lam,
          clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
          lr=args.lr, max_grad_norm=args.max_grad_norm, save_dir=args.save_dir,
          window_size=args.window_size, frame_history=args.frame_history)


# In[ ]:




