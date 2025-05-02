# Custom SAC with RND and ViT for ViZDoom Deadly Corridor
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
import random

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


# === SAC Agent ===
class SACAgent(nn.Module):
    def __init__(self, vit_encoder, action_dim, hidden_dim=1024):
        super().__init__()
        self.vit = vit_encoder
        
        # Policy network (actor)
        self.policy = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Twin Q-networks (critics)
        self.q1 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Target Q-networks
        self.target_q1 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_q2 = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize target networks with the same weights
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Freeze target networks
        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False
        
        # Log alpha (temperature parameter)
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.target_entropy = -0.98 * np.log(action_dim)  # heuristic

    def forward(self, obs):
        # Get features from ViT encoder
        feat = self.vit(obs)
        # Forward through policy network
        logits = self.policy(feat)
        return logits, feat

    def get_action(self, obs, deterministic=False):
        # Ensure we're in evaluation mode for inference
        self.eval()
        with torch.no_grad():
            logits, feat = self.forward(obs)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
                return action, None, None, feat
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Create categorical distribution
            dist = torch.distributions.Categorical(probs)
            # Sample action from distribution
            action = dist.sample()
            # Calculate log probability of the action
            log_prob = dist.log_prob(action)
            # Calculate entropy
            entropy = dist.entropy()
            
            return action, log_prob, entropy, feat
    
    def get_q_values(self, obs, eval_mode=False):
        if eval_mode:
            self.eval()
            with torch.no_grad():
                feat = self.vit(obs)
                q1 = self.q1(feat)
                q2 = self.q2(feat)
                return q1, q2, feat
        else:
            feat = self.vit(obs)
            q1 = self.q1(feat)
            q2 = self.q2(feat)
            return q1, q2, feat
    
    def get_target_q_values(self, obs):
        self.eval()
        with torch.no_grad():
            feat = self.vit(obs)
            target_q1 = self.target_q1(feat)
            target_q2 = self.target_q2(feat)
            return target_q1, target_q2
    
    def soft_update_targets(self, tau=0.005):
        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, obs, action, reward, next_obs, done, feat):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done, feat)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, feat = zip(*batch)
        
        return (
            torch.stack(obs),
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_obs),
            torch.tensor(done, dtype=torch.float32),
            torch.stack(feat)
        )
    
    def __len__(self):
        return len(self.buffer)


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
            num_spatial_blocks=2,
            num_temporal_blocks=2,
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
def save_checkpoint(path, agent, vit_encoder, rnd_model, optimizer_policy, optimizer_q, optimizer_alpha, 
                    rnd_optimizer, timesteps, rewards):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'vit_encoder_state_dict': agent.vit.state_dict(),
        'rnd_model_state_dict': rnd_model.state_dict(),
        'optimizer_policy_state_dict': optimizer_policy.state_dict(),
        'optimizer_q_state_dict': optimizer_q.state_dict(),
        'optimizer_alpha_state_dict': optimizer_alpha.state_dict(),
        'rnd_optimizer_state_dict': rnd_optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards
    }
    torch.save(checkpoint, path)


# === Training Loop ===
def train(env_id="VizdoomMyWayHome-v0", total_timesteps=500_000, batch_size=64, 
          save_dir="models", window_size=10, replay_size=100000, 
          reward_scale=1.0, learning_starts=5000, update_interval=1):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_sac_rnd_vit_best.pt")
    
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
    agent = SACAgent(vit_encoder, env.action_space.n).to(device)
    print("Models initialized on device")

    # Define reward weight as a learnable parameter
    reward_weight = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32, device=device))
    reward_weight.register_hook(lambda grad: grad.clamp_(-1, 1))  # Optional: Clamp gradient

    # Setup optimizers for SAC
    optimizer_policy = torch.optim.Adam(list(agent.policy.parameters()) + list(vit_encoder.parameters()), lr=3e-4)
    optimizer_q = torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=3e-4)
    optimizer_alpha = torch.optim.Adam([agent.log_alpha], lr=3e-4)
    optimizer_reward_weight = torch.optim.Adam([reward_weight], lr=3e-4)
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=1e-4)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=replay_size)

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
    for timestep in range(1, total_timesteps + 1):
        # Extract 'screen' from the observation dictionary
        obs_screen = obs['screen'] if isinstance(obs, dict) else obs
        
        # Convert to tensor and move to device
        obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=device)
        
        # Ensure correct shape [C, H, W]
        if len(obs_tensor.shape) == 3 and obs_tensor.shape[0] != 3:  # Assuming [H, W, C]
            obs_tensor = obs_tensor.permute(2, 0, 1)  # Change to [C, H, W]
        
        # Random actions before learning starts
        if timestep < learning_starts:
            action = torch.tensor(env.action_space.sample(), device=device)
            with torch.no_grad():
                _, _, _, feature = agent.get_action(obs_tensor.unsqueeze(0))
        else:
            # Forward pass through agent
            with torch.no_grad():
                action, _, _, feature = agent.get_action(obs_tensor.unsqueeze(0))
        
        # Execute action in environment
        next_obs, extrinsic_reward, terminated, truncated, _ = env.step(action.cpu().item())
        done = terminated or truncated
        
        # RND intrinsic reward
        with torch.no_grad():
            intrinsic_reward = rnd_model(feature).cpu().item()
        
        # Calculate total reward
        clamped_reward_weight = reward_weight.data.clamp(0.01, 0.25)
        total_reward = extrinsic_reward + clamped_reward_weight * intrinsic_reward
        
        # Prepare next observation
        next_obs_screen = next_obs['screen'] if isinstance(next_obs, dict) else next_obs
        next_obs_tensor = torch.tensor(next_obs_screen, dtype=torch.float32, device=device)
        if len(next_obs_tensor.shape) == 3 and next_obs_tensor.shape[0] != 3:
            next_obs_tensor = next_obs_tensor.permute(2, 0, 1)
        
        # Store transition in replay buffer
        replay_buffer.push(
            obs_tensor, 
            action.item(), 
            total_reward * reward_scale, 
            next_obs_tensor, 
            float(done), 
            feature.squeeze(0)
        )
        
        # Update current episode stats
        episodes_completed += int(done)
        
        # Update observation
        obs = next_obs
        
        # Reset if episode is done
        if done:
            episode_rewards.append(extrinsic_reward)  # Track only extrinsic rewards for evaluation
            obs, _ = env.reset()
            vit_encoder.reset_buffer()  # Reset the feature extractor's buffer
        
        # Training updates
        if timestep >= learning_starts and len(replay_buffer) >= batch_size:
            if timestep % update_interval == 0:
                for _ in range(update_interval):  # Multiple updates per step
                    # Sample from replay buffer
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch, feat_batch = replay_buffer.sample(batch_size)
                    
                    # Move to device
                    state_batch = state_batch.to(device)
                    action_batch = action_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    next_state_batch = next_state_batch.to(device)
                    done_batch = done_batch.to(device)
                    feat_batch = feat_batch.to(device)
                    
                    # Get current alpha value
                    alpha = torch.exp(agent.log_alpha).detach()
                    
                    # === Q-network update ===
                    with torch.no_grad():
                        # Get next action distribution
                        next_logits, _ = agent(next_state_batch)
                        next_probs = F.softmax(next_logits, dim=-1)
                        next_log_probs = F.log_softmax(next_logits, dim=-1)
                        
                        # Target Q-values with entropy regularization
                        next_target_q1, next_target_q2 = agent.get_target_q_values(next_state_batch)
                        next_q = torch.min(next_target_q1, next_target_q2)
                        
                        # Calculate expected Q-value
                        expected_q = next_probs * (next_q - alpha * next_log_probs)
                        expected_q = expected_q.sum(dim=1)
                        target_q = reward_batch + (1.0 - done_batch) * 0.99 * expected_q
                    
                    # Current Q-values
                    current_q1, current_q2, _ = agent.get_q_values(state_batch)
                    
                    # Extract Q-values for the actions that were taken
                    current_q1 = current_q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    current_q2 = current_q2.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    
                    # Compute Q-network losses
                    q1_loss = F.mse_loss(current_q1, target_q)
                    q2_loss = F.mse_loss(current_q2, target_q)
                    q_loss = q1_loss + q2_loss
                    
                    # Update Q-networks
                    optimizer_q.zero_grad()
                    q_loss.backward()
                    optimizer_q.step()
                    
                    # === Policy update ===
                    logits, _ = agent(state_batch)
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get current Q-values
                    current_q1, current_q2, _ = agent.get_q_values(state_batch)
                    min_q = torch.min(current_q1, current_q2)
                    
                    # Calculate policy loss (maximum entropy objective)
                    inside_term = alpha * log_probs - min_q
                    policy_loss = (probs * inside_term).sum(dim=1).mean()
                    
                    # Update policy network
                    optimizer_policy.zero_grad()
                    policy_loss.backward()
                    optimizer_policy.step()
                    
                    # === Alpha update ===
                    # Calculate entropy
                    entropy = -(log_probs * probs).sum(dim=1).mean()
                    
                    # Calculate alpha loss
                    alpha_loss = -agent.log_alpha * (entropy - agent.target_entropy).detach()
                    
                    # Update alpha
                    optimizer_alpha.zero_grad()
                    alpha_loss.backward()
                    optimizer_alpha.step()
                    
                    # === RND update ===
                    # Training RND using current observations
                    rnd_loss = rnd_model(feat_batch.detach()).mean()
                    
                    # Update RND predictor
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()
                    
                    # === Reward weight update ===
                    reward_weight_loss = -torch.mean(target_q.detach() * reward_weight)
                    
                    optimizer_reward_weight.zero_grad()
                    reward_weight_loss.backward()
                    optimizer_reward_weight.step()
                    
                    # Clamp reward weight after update
                    reward_weight.data.clamp_(0.01, 0.25)
                    
                    # === Target network update ===
                    agent.soft_update_targets(tau=0.005)
        
        # Reporting and model saving
        if timestep % 1000 == 0:
            # Calculate statistics
            time_elapsed = time.time() - start_time
            fps = timestep / time_elapsed
            
            # Calculate mean reward over window
            if len(episode_rewards) > 0:
                window = min(window_size, len(episode_rewards))
                recent_mean_reward = np.mean(episode_rewards[-window:])
            else:
                recent_mean_reward = float('-inf')
            
            # Report progress
            print(f"\n{'='*64}")
            print(f"Timestep: {timestep}/{total_timesteps} ({timestep/total_timesteps*100:.1f}%)")
            print(f"Episodes completed: {episodes_completed}")
            print(f"FPS: {fps:.1f}")
            if len(episode_rewards) > 0:
                print(f"Recent mean reward: {recent_mean_reward:.3f}")
                if best_mean_reward != float('-inf'):
                    print(f"Best mean reward: {best_mean_reward:.3f}")
            print(f"Current alpha: {torch.exp(agent.log_alpha).item():.4f}")
            print(f"Current reward weight: {reward_weight.item():.4f}")
            print(f"{'='*64}")
            
            # Save best model
            if recent_mean_reward > best_mean_reward and len(episode_rewards) >= window_size:
                best_mean_reward = recent_mean_reward
                save_checkpoint(
                    model_path, agent, vit_encoder, rnd_model, 
                    optimizer_policy, optimizer_q, optimizer_alpha, 
                    rnd_optimizer, timestep, episode_rewards
                )
                print(f"New best model! Mean reward: {best_mean_reward:.3f} - Saved to {model_path}")
    
    # Close environment
    env.close()
    print("\nTraining completed!")
    
    # Final statistics
    if len(episode_rewards) > 0:
        print(f"\n{'='*30} Final Statistics {'='*30}")
        print(f"Total episodes: {episodes_completed}")
        print(f"Average episode reward: {np.mean(episode_rewards):.3f}")
        print(f"Best mean reward: {best_mean_reward:.3f}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Total training time: {time.time() - start_time:.1f}s")
        print(f"Best model saved to: {model_path}")
    else:
        print("No episodes completed during training.")


if __name__ == "__main__":
    train()
