# ppo_infini_ivrl.py - Deadly Corridor Version
# Based on my_way_home/ppo_infini_ivrl.py structure

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
import itertools
import argparse

# Add project root for model imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# --- Explicit Import or Fail ---
try:
    from models.infini_vit import InfiniViT
except ImportError as e:
    print("-" * 80)
    print(f"Fatal Error: Failed to import InfiniViT from models.infini_vit.")
    print(f"Please ensure the 'models' directory is in the Python path ({ROOT})")
    print(f"and the InfiniViT class is defined correctly.")
    print(f"Original Error: {e}")
    print("-" * 80)
    raise

# =============================================================================
# Running mean/variance normaliser (intrinsic rewards)
# =============================================================================
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if not isinstance(x, np.ndarray): x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0: x = x[np.newaxis]
        if x.size == 0: return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        if not isinstance(x, np.ndarray): x = np.asarray(x, dtype=np.float64)
        if x.size == 0: return np.array([], dtype=np.float32)
        return ((x - self.mean) / np.sqrt(np.maximum(self.var, 1e-8))).astype(np.float32)

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']

# =============================================================================
# β‑schedule for mixing intrinsic advantage
# =============================================================================
def beta_schedule(step: int, total_steps: int, warmup: int = 20000, final_beta: float = 0.05):
    if step < warmup: return 1.0
    prog = max(0.0, min(1.0, (step - warmup) / max(1, total_steps - warmup)))
    return final_beta + (1.0 - final_beta) * (1.0 - prog)

# =============================================================================
# Random Network Distillation (RND) module
# =============================================================================
class RNDModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False; p.data = p.data.float()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for p in self.predictor.parameters(): p.data = p.data.float()

    def forward(self, x):
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
             raise ValueError(f"RNDModel expected input shape [Batch, {self.input_dim}], but got {x.shape}")
        x = x.float()
        with torch.no_grad(): tgt = self.target(x)
        pred = self.predictor(x)
        # Return per-element loss for GAE normalisation
        return F.mse_loss(pred, tgt, reduction="none").mean(dim=1)

# =============================================================================
# Intrinsic need modules
# =============================================================================
class NeedRewardModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, need_names=None):
        super().__init__()
        self.input_dim = input_dim
        self.need_names = need_names or ["novelty", "uncertainty", "controllability", "saliency", "goal_progress"]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, len(self.need_names))
        )
        for p in self.parameters(): p.data = p.data.float()

    def forward(self, feat, need_vals: dict):
        if feat.dim() != 2 or feat.shape[-1] != self.input_dim:
            raise ValueError(f"NeedRewardModule expected feature shape [Batch, {self.input_dim}], got {feat.shape}")
        feat = feat.float(); batch_size = feat.size(0)
        values_list = []
        for n in self.need_names:
            if n not in need_vals: raise KeyError(f"Need '{n}' not found in need_vals")
            val = need_vals[n]
            if not isinstance(val, torch.Tensor): val = torch.tensor(val, dtype=torch.float32, device=feat.device)
            val = val.to(feat.device).float()
            if val.numel() == 1 and batch_size > 0: val = val.expand(batch_size)
            elif val.shape != (batch_size,):
                 if batch_size == 1 and val.dim() == 0: val = val.unsqueeze(0)
                 else: raise ValueError(f"Need '{n}' shape mismatch. Expected ({batch_size},), got {val.shape}")
            values_list.append(val)
        values = torch.stack(values_list, dim=1); logits = self.mlp(feat)
        weights = F.softmax(logits, dim=-1)
        return (weights * values).sum(dim=1), weights # Return weighted sum and the weights themselves

class LatentNeedBank(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.need_names = ["uncertainty", "controllability", "saliency", "goal_progress"]
        self.heads = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for n in self.need_names if n != "saliency"
        })
        # Controllability predicts next state features based on current state and action
        self.ctrl = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.prev_feat = None; self.prev_goal_val = None
        for p in self.parameters(): p.data = p.data.float()

    def reset_state(self):
        self.prev_feat = None; self.prev_goal_val = None

    def forward(self, feat, action=None):
        if feat.dim() != 2 or feat.shape[-1] != self.input_dim:
             raise ValueError(f"LatentNeedBank expected feature shape [Batch, {self.input_dim}], got {feat.shape}")
        feat = feat.float(); batch_size = feat.size(0); vals = {}

        # Uncertainty (using dropout variance)
        uncertainty_head = self.heads["uncertainty"]
        if self.training:
            samples = torch.stack([uncertainty_head(F.dropout(feat, 0.2, True)).squeeze(-1) for _ in range(10)])
            vals["uncertainty"] = samples.std(dim=0)
        else:
            vals["uncertainty"] = uncertainty_head(feat).squeeze(-1)

        # Saliency (change in features)
        if self.prev_feat is not None:
            prev_feat_device = self.prev_feat.to(feat.device)
            # Handle batch size mismatch between stored state and current input
            if prev_feat_device.shape[0] != feat.shape[0]:
                # If we're processing batches but have single example state, disable state comparison
                vals["saliency"] = torch.zeros(batch_size, device=feat.device)
            else:
                vals["saliency"] = (feat - prev_feat_device).pow(2).mean(dim=1)
        else:
            vals["saliency"] = torch.zeros(batch_size, device=feat.device)

        # Controllability (predictability of next state)
        if self.prev_feat is not None and action is not None:
            prev_feat_device = self.prev_feat.to(feat.device)
            if isinstance(action, (int, float)): act_tensor = torch.full((batch_size, 1), float(action), device=feat.device, dtype=torch.float32)
            elif isinstance(action, torch.Tensor): act_tensor = action.float().view(batch_size, 1).to(feat.device)
            else: raise TypeError(f"Unsupported action type: {type(action)}")
            
            # Handle batch size mismatch
            if prev_feat_device.shape[0] != feat.shape[0]:
                vals["controllability"] = torch.zeros(batch_size, device=feat.device)
            else:
                pred = self.ctrl(torch.cat([prev_feat_device, act_tensor], dim=1))
                err = F.mse_loss(pred, feat.detach(), reduction="none").mean(dim=1)
                vals["controllability"] = F.relu(1.0 - err) # Higher score for lower prediction error
        else:
            vals["controllability"] = torch.zeros(batch_size, device=feat.device)

        # Goal Progress
        current_goal_val = self.heads["goal_progress"](feat).squeeze(-1)
        if self.prev_goal_val is not None:
            prev_goal_device = self.prev_goal_val.to(feat.device)
            
            # Handle batch size mismatch
            if prev_goal_device.shape != current_goal_val.shape:
                vals["goal_progress"] = torch.zeros_like(current_goal_val)
            else:
                prog = (current_goal_val - prev_goal_device).clamp(min=0) # Only positive progress counts
                vals["goal_progress"] = prog
        else:
            vals["goal_progress"] = torch.zeros_like(current_goal_val)
        
        # Only update state if we're processing a single example (during rollout)
        # Don't update during batch training
        if batch_size == 1:
            self.prev_feat = feat.detach()
            self.prev_goal_val = current_goal_val.detach()
            
        return vals

# =============================================================================
# PPO Agent with dual critics
# =============================================================================
class PPOAgent(nn.Module):
    def __init__(self, vit_encoder: nn.Module, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.vit = vit_encoder; feat_dim = 512
        self.policy = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.value_ext = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.value_int = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        for p in self.parameters(): p.data = p.data.float()

    def forward(self, features):
        # Expects features directly [Batch, FeatDim]
        if features.dim() != 2 or features.shape[1] != self.policy[0].in_features:
             raise ValueError(f"Agent expected features [Batch, {self.policy[0].in_features}], got {features.shape}")
        logits = self.policy(features); v_ext = self.value_ext(features).squeeze(-1); v_int = self.value_int(features).squeeze(-1)
        return logits, v_ext, v_int

    @torch.no_grad()
    def get_action(self, obs_frame, features):
        # Expects preprocessed frame [C,H,W] and corresponding features [FeatDim]
        self.eval()
        if features.dim() == 1: features = features.unsqueeze(0) # Add batch dim if needed
        # Use forward pass which handles features shape and gets values
        logits, v_ext, v_int = self.forward(features=features) # features is [1, FeatDim]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample(); logp = dist.log_prob(action); entropy = dist.entropy()
        # Return values and features squeezed to single item
        return action.squeeze(0), logp.squeeze(0), entropy.squeeze(0), v_ext.squeeze(0), v_int.squeeze(0), features.squeeze(0)

# =============================================================================
# Rollout buffer (dual critics) - Stores precomputed features
# =============================================================================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.acts, self.ext_rews, self.int_rews, self.dones = [], [], [], []
        self.logps, self.v_ext, self.v_int, self.feats = [], [], [], []
        self.adv_ext, self.adv_int, self.ret_ext, self.ret_int = None, None, None, None
        self._size = 0
        self.is_finished = False

    def store(self, a, r_ext, r_int, d, lp, ve, vi, f):
        if self.is_finished: raise RuntimeError("Buffer is finished.")
        # We store features directly, no need for obs_frames
        self.acts.append(a)
        self.ext_rews.append(r_ext)
        self.int_rews.append(r_int)
        self.dones.append(d)
        self.logps.append(lp)
        self.v_ext.append(ve.clone().detach()) # Store scalar tensor values
        self.v_int.append(vi.clone().detach())
        self.feats.append(f.clone().detach()) # Store features
        self._size += 1

    def finish_path(self, intrinsic_reward_norm: RunningMeanStd, last_v_ext=torch.tensor(0.0), last_v_int=torch.tensor(0.0), gamma=0.99, lam=0.95, device: torch.device = torch.device('cpu')):
        if self.is_finished: print("Warning: finish_path called again."); return
        if self._size == 0: print("Warning: finish_path called on empty buffer."); self.is_finished=True; return

        # Normalize intrinsic rewards
        raw_int_rews_np = np.array([r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in self.int_rews], dtype=np.float64)
        intrinsic_reward_norm.update(raw_int_rews_np)
        norm_int_rews = intrinsic_reward_norm.normalize(raw_int_rews_np)
        norm_int_rews = np.clip(norm_int_rews, -5.0, 5.0) # Clip normalized rewards

        # Convert lists to numpy arrays for calculation
        v_ext_np = torch.stack(self.v_ext).cpu().numpy() # [N]
        v_int_np = torch.stack(self.v_int).cpu().numpy() # [N]
        dones_np = np.array(self.dones, dtype=np.float32) # [N]
        ext_rews_np = np.array(self.ext_rews, dtype=np.float32) # [N]

        # GAE calculation for both critics
        N = self._size
        adv_ext_np = np.zeros(N, dtype=np.float32)
        adv_int_np = np.zeros(N, dtype=np.float32)
        gae_ext = gae_int = 0.0
        next_v_ext = last_v_ext.cpu().item() # Ensure last values are on CPU for numpy calc
        next_v_int = last_v_int.cpu().item()

        for i in reversed(range(N)):
            mask = 1.0 - dones_np[i]
            # Extrinsic
            delta_ext = ext_rews_np[i] + gamma * next_v_ext * mask - v_ext_np[i]
            gae_ext = delta_ext + gamma * lam * mask * gae_ext
            adv_ext_np[i] = gae_ext
            next_v_ext = v_ext_np[i]
            # Intrinsic
            delta_int = norm_int_rews[i] + gamma * next_v_int * mask - v_int_np[i]
            gae_int = delta_int + gamma * lam * mask * gae_int
            adv_int_np[i] = gae_int
            next_v_int = v_int_np[i]

        # Convert back to tensors and move to the correct device
        self.adv_ext = torch.from_numpy(adv_ext_np).to(device)
        self.adv_int = torch.from_numpy(adv_int_np).to(device)
        self.ret_ext = self.adv_ext + torch.stack(self.v_ext).to(device) # Ensure stacked values are also on device
        self.ret_int = self.adv_int + torch.stack(self.v_int).to(device)

        self.is_finished = True

    def get_batches(self, batch_size: int, device: torch.device):
        if not self.is_finished: raise RuntimeError("finish_path must be called before get_batches.")
        indices = np.arange(self._size)
        np.random.shuffle(indices)

        for start in range(0, self._size, batch_size):
            end = min(start + batch_size, self._size)
            batch_indices = indices[start:end]

            # Stack data for the batch and move to device
            acts_batch = torch.tensor([self.acts[i] for i in batch_indices], dtype=torch.long).to(device)
            logps_batch = torch.stack([self.logps[i] for i in batch_indices]).to(device)
            feats_batch = torch.stack([self.feats[i] for i in batch_indices]).to(device)
            adv_ext_batch = self.adv_ext[batch_indices].to(device)
            adv_int_batch = self.adv_int[batch_indices].to(device)
            ret_ext_batch = self.ret_ext[batch_indices].to(device)
            ret_int_batch = self.ret_int[batch_indices].to(device)

            yield acts_batch, logps_batch, feats_batch, adv_ext_batch, adv_int_batch, ret_ext_batch, ret_int_batch

# =============================================================================
# InfiniViT Feature Extractor Wrapper
# =============================================================================
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history: int = 32, img_size=(84, 84), vit_model_class=InfiniViT, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.frame_history = frame_history

        # Instantiate the underlying ViT model (InfiniViT)
        self.vit = vit_model_class(
            img_size=self.img_size,
            patch_size=14,
            in_channels=3,
            num_classes=512, # Output feature dimension
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

        self.buffer = deque(maxlen=self.frame_history)
        self.reset_buffer()

    def reset_buffer(self):
        empty_frame = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32, device=self.device)
        self.buffer = deque([empty_frame.clone() for _ in range(self.frame_history)], maxlen=self.frame_history)
        if hasattr(self.vit, 'reset_memory'):
            self.vit.reset_memory()
        # print("InfiniViT wrapper buffer reset") # Less verbose

    def _preprocess(self, obs_frame):
        # Handles dict/ndarray, tensor conversion, channel permute, normalization, resize
        if isinstance(obs_frame, dict):
            obs_screen = obs_frame['screen']
        else:
            obs_screen = obs_frame

        if isinstance(obs_screen, np.ndarray):
            obs_tensor = torch.tensor(obs_screen, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs_screen.to(self.device)

        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] == 3: # HWC -> CHW
            obs_tensor = obs_tensor.permute(2, 0, 1)
        elif obs_tensor.dim() != 3 or obs_tensor.shape[0] != 3:
             raise ValueError(f"Unexpected observation shape for preprocessing: {obs_tensor.shape}")

        obs_tensor = obs_tensor / 255.0
        obs_tensor = TF.resize(obs_tensor, list(self.img_size), antialias=True)
        return obs_tensor # [C, H, W]

    def forward(self, obs):
        # Process a single observation (dict or ndarray)
        processed_frame = self._preprocess(obs) # Get [C, H, W]
        # Update internal buffer
        self.buffer.append(processed_frame.clone())
        # Stack buffer into sequence [T, C, H, W]
        stacked_sequence = torch.stack(list(self.buffer), dim=0)
        # Pass sequence to InfiniViT and return features [FeatDim]
        features = self.vit(stacked_sequence)
        return features

    # Allow direct state dict access to underlying ViT
    def state_dict(self, *args, **kwargs):
        return self.vit.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.vit.load_state_dict(state_dict, *args, **kwargs)

# =============================================================================
# Checkpoint Saving/Loading
# =============================================================================
def save_checkpoint(path, agent, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, need_optimizer, timesteps, rewards, norm_state):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'vit_encoder_state_dict': agent.vit.state_dict(), # Save underlying ViT state via wrapper
        'rnd_model_state_dict': rnd_model.state_dict(),
        'need_module_state_dict': need_module.state_dict(),
        'latent_bank_state_dict': latent_bank.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rnd_optimizer_state_dict': rnd_optimizer.state_dict(),
        'need_optimizer_state_dict': need_optimizer.state_dict(),
        'timesteps': timesteps,
        'rewards': rewards,
        'norm_state': norm_state # Save intrinsic reward normalizer state
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at timestep {timesteps}")

def load_checkpoint(path, agent, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, need_optimizer, running_norm):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        # Load ViT state into the wrapper
        agent.vit.load_state_dict(checkpoint['vit_encoder_state_dict'])
        rnd_model.load_state_dict(checkpoint['rnd_model_state_dict'])
        need_module.load_state_dict(checkpoint['need_module_state_dict'])
        latent_bank.load_state_dict(checkpoint['latent_bank_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        need_optimizer.load_state_dict(checkpoint['need_optimizer_state_dict'])
        running_norm.load_state_dict(checkpoint['norm_state'])
        timesteps = checkpoint['timesteps']
        rewards = checkpoint['rewards']
        print(f"Checkpoint loaded from {path} at timestep {timesteps}")
        return timesteps, rewards
    else:
        print(f"Checkpoint file not found: {path}. Starting from scratch.")
        return 0, []

# =============================================================================
# Training Loop
# =============================================================================
def train(env_id: str = "VizdoomCorridor-v0", # Adjusted default env_id
          total_timesteps: int = 1_000_000,
          rollout_len: int = 4096,
          batch_size: int = 64,
          K_epochs: int = 4,
          gamma: float = 0.99,
          lam: float = 0.95,
          clip_range: float = 0.2,
          vf_coef: float = 0.5,
          ent_coef: float = 0.01,
          lr: float = 2.5e-4,
          rnd_lr: float = 1e-4,
          need_lr: float = 1e-4,
          max_grad_norm: float = 0.5,
          save_dir: str = "models",
          vit_frame_history: int = 32,
          warmup_beta: int = 20000,
          final_beta_value: float = 0.05,
          freeze_vit_steps: int = 100000,
          load_checkpoint_path: str = None,
          save_freq: int = 50000,
          window_size: int = 10 # For mean reward logging
         ):

    os.makedirs(save_dir, exist_ok=True)
    # Adjusted save path
    base_model_name = f"{env_id.split('-')[0]}_ppo_infini_ivrl_best.pt"
    checkpoint_save_path = os.path.join(save_dir, base_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(env_id, render_mode=None)
    print(f"Environment created: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # --- Initialize Models ---
    vit_encoder = ViTFeatureWrapper(frame_history=vit_frame_history, device=device).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
    rnd_model = RNDModel(input_dim=512).to(device)
    need_module = NeedRewardModule(input_dim=512).to(device)
    latent_bank = LatentNeedBank(input_dim=512).to(device)
    print("Models initialized on device")

    # --- Learnable Beta for intrinsic vs extrinsic mixing ---
    # Initialize at 1.0 (pure intrinsic) and optimize via gradient descent
    beta_param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=True)

    # --- Initialize Optimizers ---
    # Combine params carefully
    optimizer = torch.optim.Adam(
        itertools.chain(
            agent.policy.parameters(),
            agent.value_ext.parameters(),
            agent.value_int.parameters(),
            agent.vit.parameters(),  # Include ViT params
            [beta_param]            # Learnable mixing parameter
        ),
        lr=lr
    )
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=rnd_lr)
    need_optimizer = torch.optim.Adam(itertools.chain(need_module.parameters(),
                                                   latent_bank.parameters()),
                                    lr=need_lr)

    # --- Rollout Buffer & Normalizer ---
    buffer = RolloutBuffer()
    intrinsic_reward_norm = RunningMeanStd()

    # --- Load Checkpoint --- 
    start_timestep = 0
    episode_rewards = []
    if load_checkpoint_path:
         loaded_timesteps, loaded_rewards = load_checkpoint(
             load_checkpoint_path, agent, rnd_model, need_module, latent_bank,
             optimizer, rnd_optimizer, need_optimizer, intrinsic_reward_norm
         )
         start_timestep = loaded_timesteps
         episode_rewards = loaded_rewards # Continue tracking rewards

    # --- Training Setup ---
    obs, _ = env.reset()
    vit_encoder.reset_buffer()
    latent_bank.reset_state()

    episodes_completed = 0
    script_start_time = time.time()
    total_episodes = 0
    best_mean_reward = float('-inf')
    last_save_timestep = start_timestep

    # ============================
    # === Main Training Loop ===
    # ============================
    current_timestep = start_timestep
    while current_timestep < total_timesteps:
        iteration_start_time = time.time()
        buffer.clear()
        episode_reward_sum = 0 # Extrinsic reward sum for current episode
        episode_length = 0
        local_episodes_completed_this_iteration = 0

        # --- Freeze ViT --- 
        vit_frozen = current_timestep < freeze_vit_steps
        for param in agent.vit.parameters():
             param.requires_grad = not vit_frozen
        if vit_frozen:
            print(f"[Timestep {current_timestep}] ViT encoder frozen.")

        agent.eval()
        rnd_model.eval()
        need_module.eval()
        latent_bank.eval()

        # --- Rollout Phase --- 
        for t in range(rollout_len):
            current_timestep += 1

            # Get features from ViT wrapper using current raw observation
            with torch.no_grad():
                features = agent.vit(obs) # Get [FeatDim]

                # Get action and values using features
                action, log_prob, _, v_ext, v_int, _ = agent.get_action(obs_frame=None, features=features)

                # Get need values
                need_values = latent_bank(features.unsqueeze(0), action.item()) # Needs batch dim
                need_values = {k: v.squeeze(0) for k, v in need_values.items()} # Remove batch dim

                # Get RND reward (novelty)
                need_values["novelty"] = rnd_model(features.unsqueeze(0)).squeeze(0)

                # Get combined intrinsic reward using Need module
                intrinsic_reward, _ = need_module(features.unsqueeze(0), need_values)
                intrinsic_reward = intrinsic_reward.squeeze(0) # Remove batch dim

            # Step environment
            next_obs, extrinsic_reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated

            # Store in buffer (use scalar tensors where appropriate)
            buffer.store(action, extrinsic_reward, intrinsic_reward, done, log_prob, v_ext, v_int, features)

            obs = next_obs
            episode_reward_sum += extrinsic_reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward_sum)
                # episode_lengths.append(episode_length) # Not strictly needed if logging mean
                obs, _ = env.reset()
                vit_encoder.reset_buffer()
                latent_bank.reset_state()
                episodes_completed += 1
                local_episodes_completed_this_iteration += 1
                episode_reward_sum = 0
                episode_length = 0

            # --- Periodic Saving --- 
            if current_timestep % save_freq == 0 and current_timestep > start_timestep:
                 periodic_save_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_infini_ivrl_ts{current_timestep}.pt")
                 save_checkpoint(periodic_save_path, agent, rnd_model, need_module, latent_bank,
                                 optimizer, rnd_optimizer, need_optimizer,
                                 current_timestep, episode_rewards, intrinsic_reward_norm.state_dict())
                 last_save_timestep = current_timestep

            # Break loop if total timesteps reached during rollout
            if current_timestep >= total_timesteps:
                 break

        # --- Post-Rollout Calculation --- 
        with torch.no_grad():
            if not done:
                last_features = agent.vit(obs)
                _, last_v_ext, last_v_int = agent.forward(last_features.unsqueeze(0))
            else:
                last_v_ext = last_v_int = torch.tensor(0.0, device=device)
        buffer.finish_path(intrinsic_reward_norm, last_v_ext, last_v_int, gamma, lam, device)

        # --- Training Phase --- 
        agent.train()
        rnd_model.train()
        need_module.train()
        latent_bank.train()

        # Logging setup
        policy_losses, value_ext_losses, value_int_losses, entropy_losses = [], [], [], []
        approx_kls, clip_fractions, rnd_losses, need_losses = [], [], [], [] # Add need_loss tracking

        for epoch in range(K_epochs):
            for acts_b, logps_b, feats_b, adv_ext_b, adv_int_b, ret_ext_b, ret_int_b in buffer.get_batches(batch_size, device):
                # Compute learned Beta for advantage mixing per batch
                beta = torch.sigmoid(beta_param)

                # --- PPO Agent Update --- 
                logits, v_ext, v_int = agent(feats_b) # Use precomputed features

                # Normalize advantages (separate normalization might be better)
                adv_ext_b = (adv_ext_b - adv_ext_b.mean()) / (adv_ext_b.std() + 1e-8)
                adv_int_b = (adv_int_b - adv_int_b.mean()) / (adv_int_b.std() + 1e-8)

                # Combine advantages using beta
                combined_adv = (1 - beta) * adv_ext_b + beta * adv_int_b

                # Policy Loss
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(acts_b)
                ratio = torch.exp(new_logps - logps_b)
                clip_adv = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * combined_adv
                pg_loss = -torch.min(ratio * combined_adv, clip_adv).mean()

                # Value Losses
                v_loss_ext = F.mse_loss(v_ext, ret_ext_b)
                v_loss_int = F.mse_loss(v_int, ret_int_b)

                # Entropy Loss
                ent_loss = dist.entropy().mean()

                # Total Agent Loss
                agent_loss = pg_loss + vf_coef * (v_loss_ext + v_loss_int) - ent_coef * ent_loss

                optimizer.zero_grad()
                agent_loss.backward()
                # Clip gradients only for non-frozen ViT
                if not vit_frozen:
                     torch.nn.utils.clip_grad_norm_(agent.vit.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(itertools.chain(agent.policy.parameters(),
                                                                agent.value_ext.parameters(),
                                                                agent.value_int.parameters()), max_grad_norm)
                optimizer.step()

                # --- RND Update --- 
                rnd_loss = rnd_model(feats_b.detach()).mean() # Use detached features
                rnd_optimizer.zero_grad()
                rnd_loss.backward()
                rnd_optimizer.step()

                # --- Need Module Update --- 
                # Recompute need values on the batch features for training
                need_vals_batch = latent_bank(feats_b.detach(), acts_b)  # Detach features
                # Get novelty from RND and detach to avoid second backward through RND graph
                need_vals_batch["novelty"] = rnd_model(feats_b.detach()).detach()
                # Calculate intrinsic reward again, but keep gradients for need modules
                intrinsic_reward_pred, _ = need_module(feats_b.detach(), need_vals_batch)
                # Use intrinsic returns as target for need module prediction (simple approach)
                need_loss = F.mse_loss(intrinsic_reward_pred, ret_int_b.detach())

                need_optimizer.zero_grad()
                need_loss.backward()
                need_optimizer.step()

                # --- Logging --- 
                policy_losses.append(pg_loss.item())
                value_ext_losses.append(v_loss_ext.item())
                value_int_losses.append(v_loss_int.item())
                entropy_losses.append(ent_loss.item())
                rnd_losses.append(rnd_loss.item())
                need_losses.append(need_loss.item())
                with torch.no_grad():
                     approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                     clip_fraction = (torch.abs(ratio - 1) > clip_range).float().mean()
                     approx_kls.append(approx_kl.item())
                     clip_fractions.append(clip_fraction.item())

        # --- End of Epoch Logging --- 
        collection_time = time.time() - iteration_start_time
        fps = rollout_len / collection_time if collection_time > 0 else float('inf')

        if len(episode_rewards) > 0:
            log_mean_reward = np.mean(episode_rewards[-window_size:])
        else:
            log_mean_reward = float('nan')

        time_elapsed = int(time.time() - script_start_time)
        current_lr = optimizer.param_groups[0]['lr']
        current_rnd_lr = rnd_optimizer.param_groups[0]['lr']
        current_need_lr = need_optimizer.param_groups[0]['lr']

        # Calculate explained variance (optional, uses buffer data)
        y_pred_ext = np.array([v.cpu().numpy() for v in buffer.v_ext])
        y_true_ext = np.array([r.cpu().numpy() for r in buffer.ret_ext])
        y_var_ext = np.var(y_true_ext)
        explained_var_ext = np.nan if y_var_ext == 0 else 1 - np.var(y_true_ext - y_pred_ext) / y_var_ext

        print(f"---------------------------------")
        print(f"| Timestep                | {current_timestep:<5} |")
        print(f"| Episodes                | {episodes_completed:<5} |")
        print(f"| rollout/                |       |")
        print(f"|    ep_rew_mean (ext)    | {log_mean_reward:<5.4f} |")
        # print(f"|    ep_len_mean          | {mean_length:<5.1f} |") # If lengths tracked
        if best_mean_reward != float('-inf'):
             print(f"|    best_mean_reward     | {best_mean_reward:<5.3f} |")
        print(f"|-------------------------|-------|")
        print(f"| time/                   |       |")
        print(f"|    fps                  | {fps:<5.1f} |")
        # print(f"|    iteration            | {iteration//rollout_len + 1:<5} |") # Iteration count depends on start_timestep
        print(f"|    time_elapsed         | {time_elapsed:<5} |")
        print(f"|-------------------------|-------|")
        print(f"| train/                  |       |")
        print(f"|    approx_kl            | {np.mean(approx_kls):<5.3f} |")
        print(f"|    clip_fraction        | {np.mean(clip_fractions):<5.3f} |")
        print(f"|    policy_loss          | {np.mean(policy_losses):<5.3f} |")
        print(f"|    value_loss_ext       | {np.mean(value_ext_losses):<5.3f} |")
        print(f"|    value_loss_int       | {np.mean(value_int_losses):<5.3f} |")
        print(f"|    entropy_loss         | {np.mean(entropy_losses):<5.3f} |")
        print(f"|    rnd_loss             | {np.mean(rnd_losses):<5.3f} |")
        print(f"|    need_loss            | {np.mean(need_losses):<5.3f} |")
        print(f"|    explained_variance   | {explained_var_ext:<5.3f} |")
        print(f"|    beta (learned)       | {beta.item():<5.3f} |")
        print(f"|    learning_rate        | {current_lr:<5.5f} |")
        print(f"|    rnd_learning_rate    | {current_rnd_lr:<5.5f} |")
        print(f"|    need_learning_rate   | {current_need_lr:<5.5f} |")
        print(f"---------------------------------")

        # --- Save Best Model --- 
        if not np.isnan(log_mean_reward) and log_mean_reward > best_mean_reward:
            best_mean_reward = log_mean_reward
            save_checkpoint(checkpoint_save_path, agent, rnd_model, need_module, latent_bank,
                            optimizer, rnd_optimizer, need_optimizer,
                            current_timestep, episode_rewards, intrinsic_reward_norm.state_dict())
            print(f"*** New best model saved! Mean reward: {best_mean_reward:.3f} ***")


    # --- End of Training --- 
    env.close()
    print("Training completed!")

    # Save final model if not recently saved
    if current_timestep > last_save_timestep:
        final_save_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_infini_ivrl_final_ts{current_timestep}.pt")
        save_checkpoint(final_save_path, agent, rnd_model, need_module, latent_bank,
                        optimizer, rnd_optimizer, need_optimizer,
                        current_timestep, episode_rewards, intrinsic_reward_norm.state_dict())

    # Final statistics
    if len(episode_rewards) > 0:
        print(f"{'='*30} Final Statistics {'='*30}")
        print(f"Total episodes completed: {episodes_completed}")
        print(f"Final average episode reward (last {window_size}): {np.mean(episode_rewards[-window_size:]):.3f}")
        print(f"Best mean reward achieved: {best_mean_reward:.3f}")
        print(f"Total timesteps: {current_timestep}")
        print(f"Total training time: {time.time() - script_start_time:.1f}s")
        print(f"Best model saved to: {checkpoint_save_path}")
    else:
        print("No complete episodes recorded during training.")

# =============================================================================
# Argument Parser and Main Execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO InfiniViT+IVRL for Deadly Corridor")
    parser.add_argument("--env_id", type=str, default="VizdoomCorridor-v0", help="Environment ID")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total timesteps for training")
    parser.add_argument("--rollout_len", type=int, default=4096, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--K_epochs", type=int, default=4, help="Number of training epochs per rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate for main optimizer")
    parser.add_argument("--rnd_lr", type=float, default=1e-4, help="Learning rate for RND optimizer")
    parser.add_argument("--need_lr", type=float, default=1e-4, help="Learning rate for Need modules optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--vit_frame_history", type=int, default=32, help="Frame history for ViT wrapper")
    parser.add_argument("--warmup_beta", type=int, default=20000, help="Timesteps before beta starts decreasing")
    parser.add_argument("--final_beta_value", type=float, default=0.05, help="Final value for beta schedule")
    parser.add_argument("--freeze_vit_steps", type=int, default=100000, help="Freeze ViT for initial steps")
    parser.add_argument("--load_checkpoint_path", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency to save checkpoints (timesteps)")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for logging mean reward")

    args = parser.parse_args()

    train(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        rollout_len=args.rollout_len,
        batch_size=args.batch_size,
        K_epochs=args.K_epochs,
        gamma=args.gamma,
        lam=args.lam,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        lr=args.lr,
        rnd_lr=args.rnd_lr,
        need_lr=args.need_lr,
        max_grad_norm=args.max_grad_norm,
        save_dir=args.save_dir,
        vit_frame_history=args.vit_frame_history,
        warmup_beta=args.warmup_beta,
        final_beta_value=args.final_beta_value,
        freeze_vit_steps=args.freeze_vit_steps,
        load_checkpoint_path=args.load_checkpoint_path,
        save_freq=args.save_freq,
        window_size=args.window_size
    )
