# Custom PPO with RND and InfiniViT for ViZDoom Deadly Corridor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import torchvision.transforms.functional as TF
from vizdoom import gymnasium_wrapper
import sys
import os
import time

# === Import InfiniViT Model ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.infini_vit import InfiniViT

# === RND Module ===
class RNDModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Freeze target network parameters
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        # Return MSE loss per element in the batch
        return F.mse_loss(pred_out, target_out, reduction="none").mean(dim=1)

# === PPO Agent ===
class PPOAgent(nn.Module):
    def __init__(self, vit_encoder, action_dim, hidden_dim=256):
        super().__init__()
        self.vit = vit_encoder # Should be the ViTFeatureWrapper instance
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

    def forward(self, obs_frame):
        # Pass single frame to wrapper, which handles sequence and gets features
        feat = self.vit(obs_frame)
        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value.squeeze(-1), feat

    def get_action(self, obs_frame):
        # obs_frame should be preprocessed: [C, H, W]
        self.eval()
        with torch.no_grad():
            # Pass single frame to forward
            logits, value, feat = self.forward(obs_frame)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value, feat

# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.obs_frames, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values, self.features = [], [], [], []
        self.advantages, self.returns = [], []

    def store(self, obs_frame, action, reward, done, log_prob, value, feature):
        self.obs_frames.append(obs_frame) # Store the processed frame [C, H, W]
        self.actions.append(action)
        self.rewards.append(reward) # Store combined reward (extrinsic + intrinsic)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value.clone().detach())
        self.features.append(feature.clone().detach()) # Store features computed by InfiniViT

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

    def get_batches(self, batch_size):
        n = len(self.obs_frames)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            yield (
                # obs_batch (not needed if using precomputed features)
                torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long),
                torch.tensor([self.log_probs[i] for i in batch_indices]),
                torch.stack([self.advantages[i] for i in batch_indices]),
                torch.stack([self.returns[i] for i in batch_indices]),
                torch.stack([self.features[i] for i in batch_indices]) # Use stored features
            )

    def clear(self):
        self.__init__()

# === InfiniViT Feature Extractor Wrapper ===
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history=32, img_size=(84, 84)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.frame_history = frame_history

        self.vit = InfiniViT(
            img_size=self.img_size,
            patch_size=14,
            in_channels=3,
            num_classes=512,
            embed_dim=512,
            num_heads=4,
            mlp_ratio=2.0,
            memory_size=256,
            window_size=32,
            dropout=0.1,
            pad_if_needed=True,
            device=self.device,
            num_spatial_blocks=3,
            num_temporal_blocks=3,
            update_interval=5,
        ).to(self.device)

        self.buffer = deque(maxlen=frame_history)
        self.reset_buffer()

    def reset_buffer(self):
        empty_frame = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32, device=self.device)
        self.buffer = deque([empty_frame.clone() for _ in range(self.frame_history)], maxlen=self.frame_history)
        if hasattr(self.vit, 'reset_memory'):
            self.vit.reset_memory()
        print("InfiniViT buffer reset")

    def process_observation(self, obs):
        if isinstance(obs, dict):
            obs_screen = obs['screen']
        else:
            obs_screen = obs
        if isinstance(obs_screen, np.ndarray):
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs_screen.to(self.device)
        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] == 3:
            obs_tensor = obs_tensor.permute(2, 0, 1)
        obs_tensor = obs_tensor / 255.0
        obs_tensor = TF.resize(obs_tensor, list(self.img_size), antialias=True)
        return obs_tensor

    def forward(self, obs_frame):
        self.buffer.append(obs_frame.clone())
        stacked = torch.stack(list(self.buffer), dim=0)
        features = self.vit(stacked)
        return features

# Function to save model checkpoints
def save_checkpoint(path, agent, rnd_model, optimizer, rnd_optimizer, timesteps, rewards):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'vit_encoder_state_dict': agent.vit.state_dict(), # Save wrapper's ViT state
        'rnd_model_state_dict': rnd_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rnd_optimizer_state_dict': rnd_optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards
    }
    torch.save(checkpoint, path)

# === Training Loop ===
def train(env_id="VizdoomCorridor-v0", total_timesteps=1_000_000, rollout_len=4096, batch_size=64,
          K_epochs=10, gamma=0.99, lam=0.95, clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
          lr=2.5e-4, rnd_lr=1e-4, max_grad_norm=0.5, initial_reward_weight=0.125, clamp_reward_weight=True,
          save_dir="models", window_size=10, frame_history=32):

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_infini_rnd_vit_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    vit_encoder = ViTFeatureWrapper(frame_history=frame_history).to(device)
    rnd_model = RNDModel(input_dim=512).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
    print("Models initialized on device")

    reward_weight = torch.nn.Parameter(torch.tensor(initial_reward_weight, dtype=torch.float32, device=device))
    if clamp_reward_weight:
         reward_weight.register_hook(lambda grad: grad.clamp_(-1, 1)) # Optional gradient clamping

    optimizer = torch.optim.Adam(list(agent.parameters()) + [reward_weight], lr=lr)
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=rnd_lr)

    buffer = RolloutBuffer()

    obs, _ = env.reset()
    vit_encoder.reset_buffer()
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
        rnd_model.eval()

        for t in range(rollout_len):
            with torch.no_grad():
                action, log_prob, _, value, feature = agent.get_action(current_obs_frame)
                intrinsic_reward = rnd_model(feature).item() # RND reward based on current feature

            next_obs, extrinsic_reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            current_reward_weight_val = reward_weight.data.item()
            if clamp_reward_weight:
                 current_reward_weight_val = np.clip(current_reward_weight_val, 0.01, 0.25)

            total_reward = extrinsic_reward + current_reward_weight_val * intrinsic_reward
            episode_reward += extrinsic_reward # Track extrinsic reward for logging
            episode_length += 1

            buffer.store(current_obs_frame, action.item(), total_reward, done, log_prob.item(), value, feature)

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                vit_encoder.reset_buffer()
                current_obs_frame = vit_encoder.process_observation(obs)
                local_episodes_completed += 1
                episode_reward = 0
                episode_length = 0
            else:
                current_obs_frame = vit_encoder.process_observation(next_obs)
                obs = next_obs

        total_episodes += local_episodes_completed
        episodes_completed += local_episodes_completed

        buffer.finish_path(gamma=gamma, lam=lam)

        agent.train()
        rnd_model.train()

        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time if collection_time > 0 else float('inf')
        current_timestep = iteration + rollout_len

        if len(episode_rewards) > 0:
            window = min(window_size, len(episode_rewards))
            mean_reward = np.mean(episode_rewards[-window:]) # Log extrinsic reward mean
            mean_length = np.mean(episode_lengths[-window:])
        else:
            mean_reward = float('nan')
            mean_length = float('nan')

        time_elapsed = int(time.time() - script_start_time)
        current_lr = optimizer.param_groups[0]['lr']
        current_rnd_lr = rnd_optimizer.param_groups[0]['lr']

        print(f"---------------------------------")
        print(f"| rollout/                |       |")
        print(f"|    ep_len_mean          | {mean_length:<5.1f} |")
        print(f"|    ep_rew_mean          | {mean_reward:<5.3f} |") # Extrinsic reward
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
            save_checkpoint(model_path, agent, rnd_model, optimizer, rnd_optimizer, current_timestep, episode_rewards)
            print(f"New best model! Mean reward: {mean_reward:.3f}. Saving to {model_path}")

        policy_losses, value_losses, entropy_losses, approx_kls, clip_fractions, rnd_losses = [], [], [], [], [], []

        for epoch in range(K_epochs):
            for act_batch, logp_batch, adv_batch, ret_batch, feat_batch in buffer.get_batches(batch_size):
                act_batch = act_batch.to(device)
                logp_batch = logp_batch.to(device)
                adv_batch = adv_batch.to(device)
                ret_batch = ret_batch.to(device)
                feat_batch = feat_batch.to(device)

                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                # PPO Agent forward pass using precomputed features
                logits = agent.policy(feat_batch)
                values = agent.value(feat_batch).squeeze(-1)

                # --- PPO Loss --- 
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(act_batch)

                ratio = torch.exp(new_log_probs - logp_batch)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()

                clip_adv = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_batch
                pg_loss = -torch.min(ratio * adv_batch, clip_adv).mean()

                v_loss = F.mse_loss(values, ret_batch)
                ent_loss = dist.entropy().mean()

                # --- Reward Weight Loss (learnable intrinsic reward coefficient) ---
                # Detach returns to only update the weight
                reward_weight_loss = -torch.mean(ret_batch.detach() * reward_weight)

                # --- Combined PPO Loss --- 
                total_agent_loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss + reward_weight_loss

                # --- RND Loss --- 
                # Detach features to only update RND predictor
                rnd_loss = rnd_model(feat_batch.detach()).mean()

                # --- Optimizers --- 
                optimizer.zero_grad()
                total_agent_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step() # Updates agent policy/value nets AND reward_weight

                rnd_optimizer.zero_grad()
                rnd_loss.backward()
                rnd_optimizer.step() # Updates RND predictor net

                # Clamp reward weight *after* optimizer step if desired
                if clamp_reward_weight:
                     reward_weight.data.clamp_(0.01, 0.25)

                # Record losses
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(ent_loss.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())
                rnd_losses.append(rnd_loss.item())

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
        print(f"|    rnd_loss             | {np.mean(rnd_losses):<5.3f} |")
        print(f"|    reward_weight        | {reward_weight.data.item():<5.3f} |") # Log current weight
        print(f"---------------------------------")

    env.close()
    print("Training completed!")

    if len(episode_rewards) > 0:
        print(f"{'='*30} Final Statistics {'='*30}")
        print(f"Total episodes: {total_episodes}")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f}") # Extrinsic reward
        print(f"Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"Best mean reward: {best_mean_reward:.3f}")
        print(f"Total timesteps: {current_timestep}")
        print(f"Total training time: {time.time() - script_start_time:.1f}s")
        print(f"Best model saved to: {model_path}")
    else:
        print("No episodes completed during training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO InfiniViT+RND for Deadly Corridor")
    parser.add_argument("--env_id", type=str, default="VizdoomCorridor-v0")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--rnd_lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--initial_reward_weight", type=float, default=0.125)
    parser.add_argument("--clamp_reward_weight", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--frame_history", type=int, default=32)
    args = parser.parse_args()

    train(env_id=args.env_id, total_timesteps=args.total_timesteps,
          rollout_len=args.rollout_len, batch_size=args.batch_size,
          K_epochs=args.K_epochs, gamma=args.gamma, lam=args.lam,
          clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
          lr=args.lr, rnd_lr=args.rnd_lr, max_grad_norm=args.max_grad_norm,
          initial_reward_weight=args.initial_reward_weight, clamp_reward_weight=args.clamp_reward_weight,
          save_dir=args.save_dir, window_size=args.window_size, frame_history=args.frame_history)
