# Custom PPO with RND and ViT for ViZDoom Deadly Corridor (Full Version with Epoch Tracking)
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

# === Import your ViT Model ===
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
        return F.mse_loss(pred_out, target_out, reduction="none").mean(dim=1)


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
        # Clone and detach value before storing to prevent potential side effects
        self.values.append(value.clone().detach())
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
        for start in range(0, n, batch_size):
            end = start + batch_size
            yield [
                torch.stack([self.obs[i] for i in indices[start:end]]),
                torch.tensor([self.actions[i] for i in indices[start:end]]),
                torch.tensor([self.log_probs[i] for i in indices[start:end]]),
                torch.stack([self.advantages[i] for i in indices[start:end]]),
                torch.stack([self.returns[i] for i in indices[start:end]]),
                torch.stack([self.features[i] for i in indices[start:end]])
            ]

    def clear(self):
        self.__init__()


# === Preprocessing ViT Feature Extractor ===
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history=32):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = InfiniViT(
            img_size=(84, 84),
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
        )
        self.frame_history = frame_history
        self.buffer = deque(maxlen=frame_history)
        # Initialize buffer with empty frames
        self.reset_buffer()
        
    def reset_buffer(self):
        """Initialize the buffer with zero frames"""
        empty_frame = torch.zeros((3, 84, 84), dtype=torch.float32, device=self.device)
        self.buffer = deque([empty_frame.clone() for _ in range(self.frame_history)], maxlen=self.frame_history)

    def forward(self, obs):
        # Handle both dictionary observations and direct arrays
        if isinstance(obs, dict):
            obs = obs['screen']
        
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        # Ensure correct dimensionality and normalization
        obs = obs / 255.0
        obs = TF.resize(obs, (84, 84), antialias=True)
        
        # Check if this is a batch of observations or a single observation
        if len(obs.shape) == 3:  # Single observation [C, H, W]
            # Add to buffer and maintain buffer size
            self.buffer.append(obs.clone())
            # Stack frames into a sequence [T, C, H, W]
            stacked = torch.stack(list(self.buffer), dim=0)
            # Pass to ViT and get features
            # InfiniViT expects [T, C, H, W] for a single sequence
            return self.vit(stacked)
        else:  # Batch of observations [B, C, H, W]
            batch_size = obs.shape[0]
            features = []
            # Process each observation in the batch individually
            for i in range(batch_size):
                self.buffer.append(obs[i].clone())
                stacked = torch.stack(list(self.buffer), dim=0)  # [T, C, H, W]
                feature = self.vit(stacked)
                features.append(feature)
            # Stack features from batch
            return torch.stack(features, dim=0)  # [B, F]


# Function to save model checkpoints
def save_checkpoint(path, agent, vit_encoder, rnd_model, optimizer, rnd_optimizer, timesteps, rewards):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'vit_encoder_state_dict': agent.vit.state_dict(),
        'rnd_model_state_dict': rnd_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rnd_optimizer_state_dict': rnd_optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards
    }
    torch.save(checkpoint, path)


# === Training Loop ===
def train(env_id="VizdoomMyWayHome-v0", total_timesteps=500_000, rollout_len=4096, batch_size=64, K_epochs=10,
          save_dir="models", window_size=10):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_rnd_vit_best.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Initialize models
    vit_encoder = ViTFeatureWrapper().to(device)
    rnd_model = RNDModel(input_dim=512).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
    print("Models initialized on device")

    # Define reward weight as a learnable parameter
    reward_weight = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32, device=device))
    reward_weight.register_hook(lambda grad: grad.clamp_(-1, 1)) # Optional: Clamp gradient

    # Include reward_weight in the main optimizer
    optimizer = torch.optim.Adam(list(agent.parameters()) + list(vit_encoder.parameters()) + [reward_weight], lr=2.5e-4)
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=1e-4)

    buffer = RolloutBuffer()

    obs, _ = env.reset()
    
    # Initialize metrics tracking
    episode_rewards = []
    episode_lengths = []
    episodes_completed = 0
    start_time = time.time()
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
        
        # Collection phase
        print(f"\n{'='*20} Iteration {iteration//rollout_len + 1} - Collecting experiences {'='*20}")
        for t in range(rollout_len):
            # Extract 'screen' from the observation dictionary
            obs_screen = obs['screen'] if isinstance(obs, dict) else obs
            
            # Convert to tensor and move to device
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=device)
            
            # Ensure correct shape [C, H, W]
            if len(obs_tensor.shape) == 3 and obs_tensor.shape[0] != 3:  # Assuming [H, W, C]
                obs_tensor = obs_tensor.permute(2, 0, 1)  # Change to [C, H, W]
                
            # Forward pass through agent
            with torch.no_grad():
                action, log_prob, _, value, feature = agent.get_action(obs_tensor.unsqueeze(0))
                intrinsic_reward = rnd_model(feature).cpu().item()

            # Execute action in environment
            next_obs, extrinsic_reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            
            # Calculate total reward using the learnable weight (parameter defined outside loop)
            # Clamp the weight during calculation to ensure it stays within bounds for reward scaling
            clamped_reward_weight = reward_weight.data.clamp(0.01, 0.25)
            total_reward = extrinsic_reward + clamped_reward_weight * intrinsic_reward
            episode_reward += extrinsic_reward
            episode_length += 1

            # Store transition in buffer
            buffer.store(obs_tensor, action.item(), total_reward, done, log_prob.item(), value, feature.squeeze(0))
            
            # Update observation
            obs = next_obs

            # Reset if episode is done
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                local_episodes_completed += 1
                episode_reward = 0
                episode_length = 0
                # Reset the feature extractor's buffer to avoid contamination between episodes
                vit_encoder.reset_buffer()

        # Update total episodes
        total_episodes += local_episodes_completed
        episodes_completed += local_episodes_completed
        
        # Calculate advantages and returns
        buffer.finish_path()
        
        # Training phase - set to training mode
        agent.train()
        
        # Print collection statistics
        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time
        
        current_timestep = iteration + rollout_len
        
        # Calculate mean reward over the most recent episodes
        if len(episode_rewards) > 0:
            window = min(window_size, len(episode_rewards))
            recent_mean_reward = np.mean(episode_rewards[-window:])
            recent_mean_length = np.mean(episode_lengths[-window:])
        else:
            recent_mean_reward = float('-inf')
            recent_mean_length = 0
            
        print(f"\n{'='*20} Training on collected experiences {'='*20}")
        print(f"{'='*64}")
        print(f"| {'rollout/':20} | {'':40} |")
        print(f"| {'fps':20} | {fps:40.1f} |")
        if len(episode_rewards) > 0:
            print(f"| {'ep_len_mean':20} | {recent_mean_length:40.1f} |")
            print(f"| {'ep_rew_mean':20} | {recent_mean_reward:40.3f} |")
            if best_mean_reward != float('-inf'):
                print(f"| {'best_reward':20} | {best_mean_reward:40.3f} |")
        print(f"| {'episodes':20} | {local_episodes_completed:40d} |")
        print(f"| {'timesteps_so_far':20} | {current_timestep:40d} |")
        print(f"{'='*64}")
        
        # Save the best model based on mean reward
        if recent_mean_reward > best_mean_reward and len(episode_rewards) >= window_size:
            best_mean_reward = recent_mean_reward
            save_checkpoint(model_path, agent, vit_encoder, rnd_model, optimizer, rnd_optimizer, current_timestep, episode_rewards)
            print(f"New best model! Mean reward: {best_mean_reward:.3f} - Saved to {model_path}")
        
        # Perform multiple epochs of training
        epoch_start_time = time.time()
        for epoch in range(K_epochs):
            policy_losses, value_losses, entropy_losses, rnd_losses = [], [], [], []
            
            # Train on minibatches
            for obs_batch, act_batch, logp_batch, adv_batch, ret_batch, feat_batch in buffer.get_batches(batch_size):
                # Move data to device
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                logp_batch = logp_batch.to(device)
                adv_batch = adv_batch.to(device)
                ret_batch = ret_batch.to(device)
                feat_batch = feat_batch.to(device)
                
                # Normalize advantages
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
                
                # Forward pass
                logits, values, _ = agent(obs_batch)
                
                # Calculate policy loss
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(act_batch)
                
                # Calculate KL divergence (for monitoring)
                ratio = torch.exp(new_log_probs - logp_batch)
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                
                # Clip fraction (for monitoring)
                clip_fraction = (torch.abs(ratio - 1) > 0.2).float().mean().item()
                
                # PPO objective
                ratio = torch.exp(new_log_probs - logp_batch)
                clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv_batch
                policy_loss = -torch.min(ratio * adv_batch, clip_adv).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, ret_batch)
                
                # Entropy bonus
                entropy_loss = dist.entropy().mean()
                
                # Total loss for agent
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                # Backpropagate agent loss (calculates grads for agent + vit)
                optimizer.zero_grad()
                rnd_optimizer.zero_grad()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                # Optimizer step for agent+vit+reward_weight will happen after all grads are computed
                
                # RND update - learning from surprise
                # Detach feat_batch to prevent gradients flowing back to ViT from RND loss
                rnd_loss = rnd_model(feat_batch.detach()).mean()
                # Backpropagate RND loss (calculates grads for rnd_model.predictor)
                rnd_loss.backward()
                # Optimizer step for RND predictor will happen later
                
                # Update reward weight
                # Detach ret_batch to prevent gradients flowing back to value network
                reward_weight_loss = -torch.mean(ret_batch.detach() * reward_weight)
                # Backpropagate reward weight loss (calculates grads for reward_weight)
                reward_weight_loss.backward()
                # Optimizer step for reward_weight will happen with the main optimizer step
                
                # Step optimizers
                optimizer.step() # Updates agent, vit, and reward_weight
                rnd_optimizer.step() # Updates RND predictor

                # Clamp reward weight *after* optimizer step
                reward_weight.data.clamp_(0.01, 0.25)  # Ensure reward weight stays in bounds
                
                # Record losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                rnd_losses.append(rnd_loss.item())
                
            # Calculate explained variance
            y_pred = np.array([v.detach().cpu().numpy() for v in buffer.values])
            y_true = np.array([r.detach().cpu().numpy() for r in buffer.returns])
            y_var = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (y_var + 1e-8)
            
            # Print epoch statistics in SB3 style
            if epoch == K_epochs - 1:  # Only print on the last epoch
                total_time = time.time() - start_time
                epoch_time = time.time() - epoch_start_time
                
                print(f"\n{'='*64}")
                print(f"| {'time/':20} | {'':40} |")
                print(f"| {'fps':20} | {current_timestep / total_time:40.1f} |")
                print(f"| {'iterations':20} | {iteration//rollout_len + 1:40d} |")
                print(f"| {'epoch_time_s':20} | {epoch_time:40.1f} |")
                print(f"| {'total_timesteps':20} | {current_timestep:40d} |")
                print(f"| {'total_time_s':20} | {total_time:40.1f} |")
                print(f"{'='*64}")
                print(f"| {'train/':20} | {'':40} |")
                print(f"| {'approx_kl':20} | {approx_kl:40.6f} |")
                print(f"| {'clip_fraction':20} | {clip_fraction:40.3f} |")
                print(f"| {'clip_range':20} | {0.2:40.1f} |")
                print(f"| {'entropy_loss':20} | {np.mean(entropy_losses):40.3f} |")
                print(f"| {'explained_variance':20} | {explained_var:40.3f} |")
                print(f"| {'learning_rate':20} | {2.5e-4:40.4f} |")
                print(f"| {'policy_loss':20} | {np.mean(policy_losses):40.4f} |")
                print(f"| {'value_loss':20} | {np.mean(value_losses):40.4f} |")
                print(f"| {'rnd_loss':20} | {np.mean(rnd_losses):40.4f} |")
                print(f"{'='*64}")
                
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
        print(f"Total training time: {time.time() - start_time:.1f}s")
        print(f"Best model saved to: {model_path}")
    else:
        print("No episodes completed during training.")


if __name__ == "__main__":
    train()
