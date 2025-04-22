# ppo_infini_ivrl.py — Minimally modified version with key fixes:
# - Correct dual critic GAE calculation with separate rewards
# - RND integration for novelty
# - Learnable Need Modules
# - Basic shape handling fix
# - Fixed dict observation handling in get_action
# - Removed TensorBoard

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
import itertools # For combining parameters for optimizer

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
        return (weights * values).sum(dim=1), weights

class LatentNeedBank(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.need_names = ["uncertainty", "controllability", "saliency", "goal_progress"]
        self.heads = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for n in self.need_names if n != "saliency"
        })
        self.ctrl = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.prev_feat = None; self.prev_goal_val = None
        for p in self.parameters(): p.data = p.data.float()

    def reset_state(self):
        self.prev_feat = None; self.prev_goal_val = None

    def forward(self, feat, action=None):
        if feat.dim() != 2 or feat.shape[-1] != self.input_dim:
             raise ValueError(f"LatentNeedBank expected feature shape [Batch, {self.input_dim}], got {feat.shape}")
        feat = feat.float(); batch_size = feat.size(0); vals = {}
        uncertainty_head = self.heads["uncertainty"]
        if self.training:
            samples = torch.stack([uncertainty_head(F.dropout(feat, 0.2, True)).squeeze(-1) for _ in range(10)])
            vals["uncertainty"] = samples.std(dim=0)
        else: vals["uncertainty"] = uncertainty_head(feat).squeeze(-1)
        if self.prev_feat is not None:
            prev_feat_device = self.prev_feat.to(feat.device)
            if prev_feat_device.shape != feat.shape: raise ValueError(f"Saliency shape mismatch: {prev_feat_device.shape} vs {feat.shape}")
            vals["saliency"] = (feat - prev_feat_device).pow(2).mean(dim=1)
        else: vals["saliency"] = torch.zeros(batch_size, device=feat.device)
        if self.prev_feat is not None and action is not None:
            prev_feat_device = self.prev_feat.to(feat.device)
            if isinstance(action, (int, float)): act_tensor = torch.full((batch_size, 1), float(action), device=feat.device, dtype=torch.float32)
            elif isinstance(action, torch.Tensor): act_tensor = action.float().view(batch_size, 1).to(feat.device)
            else: raise TypeError(f"Unsupported action type: {type(action)}")
            if prev_feat_device.shape != feat.shape: raise ValueError(f"Controllability shape mismatch: {prev_feat_device.shape} vs {feat.shape}")
            if act_tensor.shape[0] != batch_size: raise ValueError(f"Controllability action batch mismatch: {act_tensor.shape[0]} vs {batch_size}")
            pred = self.ctrl(torch.cat([prev_feat_device, act_tensor], dim=1))
            err = F.mse_loss(pred, feat.detach(), reduction="none").mean(dim=1)
            vals["controllability"] = F.relu(1.0 - err)
        else: vals["controllability"] = torch.zeros(batch_size, device=feat.device)
        current_goal_val = self.heads["goal_progress"](feat).squeeze(-1)
        if self.prev_goal_val is not None:
            prev_goal_device = self.prev_goal_val.to(feat.device)
            if prev_goal_device.shape != current_goal_val.shape: raise ValueError(f"Goal progress shape mismatch: {prev_goal_device.shape} vs {current_goal_val.shape}")
            prog = (current_goal_val - prev_goal_device).clamp(min=0)
            vals["goal_progress"] = prog; self.prev_goal_val = current_goal_val.detach()
        else: vals["goal_progress"] = torch.zeros_like(current_goal_val); self.prev_goal_val = current_goal_val.detach()
        self.prev_feat = feat.detach()
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

    def forward(self, obs=None, features=None):
        if features is None:
            if obs is None: raise ValueError("Agent forward requires obs or features")
            # Pass raw obs (e.g., dict) to ViT wrapper
            features = self.vit(obs) # Wrapper outputs [FeatureDim] or [Batch, FeatureDim]
        if features.dim() == 1:
            if features.shape[0] == self.policy[0].in_features: features = features.unsqueeze(0)
            else: raise ValueError(f"Agent got 1D features with wrong dim: {features.shape}")
        elif features.dim() != 2 or features.shape[1] != self.policy[0].in_features:
             raise ValueError(f"Agent expected features [Batch, {self.policy[0].in_features}], got {features.shape}")
        logits = self.policy(features); v_ext = self.value_ext(features).squeeze(-1); v_int = self.value_int(features).squeeze(-1)
        return logits, v_ext, v_int, features

    @torch.no_grad()
    def get_action(self, obs):
        # Input 'obs' can be dict or ndarray
        self.eval()
        # *** REMOVED obs = obs.float() ***
        # Pass raw obs to ViT wrapper, get features [FeatureDim] or [1, FeatureDim]
        features_raw = self.vit(obs)
        # Use forward pass which handles features shape and gets values
        logits, v_ext, v_int, features = self.forward(obs=None, features=features_raw) # features is [1, FeatDim]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample(); logp = dist.log_prob(action); entropy = dist.entropy()
        return action, logp, entropy, v_ext, v_int, features

# =============================================================================
# Rollout buffer (dual critics) - Modified for separate rewards GAE
# =============================================================================
class RolloutBuffer:
    def __init__(self): self.clear()
    def clear(self):
        self.obs, self.acts, self.ext_rews, self.int_rews, self.dones = [], [], [], [], []
        self.logps, self.v_ext, self.v_int, self.feats = [], [], [], []
        self.adv_ext, self.adv_int, self.ret_ext, self.ret_int = None, None, None, None
        self._size = 0; self.is_finished = False
    def store(self, o, a, r_ext, r_int, d, lp, ve, vi, f):
        if self.is_finished: raise RuntimeError("Buffer is finished.")
        self.obs.append(o); self.acts.append(a); self.ext_rews.append(r_ext); self.int_rews.append(r_int)
        self.dones.append(d); self.logps.append(lp); self.v_ext.append(ve); self.v_int.append(vi)
        self.feats.append(f); self._size += 1
    def finish_path(self, intrinsic_reward_norm: RunningMeanStd, last_v_ext=0.0, last_v_int=0.0, gamma=0.99, lam=0.95):
        if self.is_finished: print("Warning: finish_path called again."); return
        if self._size == 0: print("Warning: finish_path called on empty buffer."); self.is_finished=True; return
        raw_int_rews_np = np.array(self.int_rews, dtype=np.float64)
        norm_int_rews = intrinsic_reward_norm.normalize(raw_int_rews_np)
        norm_int_rews = np.clip(norm_int_rews, -1.0, 1.0)
        v_ext_np = np.array(self.v_ext, dtype=np.float32); v_int_np = np.array(self.v_int, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32); ext_rews_np = np.array(self.ext_rews, dtype=np.float32)
        N = self._size; self.adv_ext = np.zeros(N, dtype=np.float32); self.adv_int = np.zeros(N, dtype=np.float32)
        gaeE = gaeI = 0.0; next_v_ext, next_v_int = last_v_ext, last_v_int
        for i in reversed(range(N)):
            deltaE = ext_rews_np[i] + gamma * next_v_ext * (1 - dones_np[i]) - v_ext_np[i]
            current_norm_int_rew = norm_int_rews[i] if i < len(norm_int_rews) else 0.0
            deltaI = current_norm_int_rew + gamma * next_v_int * (1 - dones_np[i]) - v_int_np[i]
            gaeE = deltaE + gamma * lam * (1 - dones_np[i]) * gaeE
            gaeI = deltaI + gamma * lam * (1 - dones_np[i]) * gaeI
            self.adv_ext[i] = gaeE; self.adv_int[i] = gaeI
            next_v_ext, next_v_int = v_ext_np[i], v_int_np[i]
        self.ret_ext = self.adv_ext + v_ext_np; self.ret_int = self.adv_int + v_int_np
        self.is_finished = True
    def get_batches(self, batch_size: int, device: torch.device):
        if not self.is_finished or self._size == 0 or self.adv_ext is None: return iter([])
        n_samples = self._size; indices = np.random.permutation(n_samples)
        try:
            all_obs = torch.stack(self.obs).to(device); all_acts = torch.tensor(self.acts, dtype=torch.long, device=device)
            all_logps = torch.tensor(self.logps, dtype=torch.float32, device=device)
            all_adv_ext = torch.tensor(self.adv_ext, dtype=torch.float32, device=device)
            all_adv_int = torch.tensor(self.adv_int, dtype=torch.float32, device=device)
            all_ret_ext = torch.tensor(self.ret_ext, dtype=torch.float32, device=device)
            all_ret_int = torch.tensor(self.ret_int, dtype=torch.float32, device=device)
            all_feats = torch.stack(self.feats).to(device)
        except Exception as e: raise RuntimeError(f"Failed to convert buffer data to tensors: {e}")
        for start in range(0, n_samples, batch_size):
            end = start + batch_size; batch_indices = indices[start:end]
            yield (all_obs[batch_indices], all_acts[batch_indices], all_logps[batch_indices],
                   all_adv_ext[batch_indices], all_adv_int[batch_indices], all_ret_ext[batch_indices],
                   all_ret_int[batch_indices], all_feats[batch_indices])

# =============================================================================
# ViT feature extractor wrapper
# =============================================================================
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history: int = 32, img_size=(84, 84), vit_model_class=InfiniViT, device=None):
        super().__init__()
        if not issubclass(vit_model_class, nn.Module): raise TypeError(f"vit_model_class must be a torch.nn.Module subclass")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size; self.frame_history = frame_history
        self.vit = vit_model_class(img_size=self.img_size, patch_size=14, in_channels=3, num_classes=512, embed_dim=512,
                                   num_heads=4, mlp_ratio=2.0, memory_size=256, window_size=32, dropout=0.1, pad_if_needed=True,
                                   device=self.device, num_spatial_blocks=3, num_temporal_blocks=3, update_interval=5
                                  ).to(self.device).float()
        self.buffer = deque(maxlen=self.frame_history); self.reset_buffer()
    def reset_buffer(self):
        empty = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32, device=self.device)
        self.buffer = deque([empty.clone() for _ in range(self.frame_history)], maxlen=self.frame_history)
    def _preprocess(self, obs_frame):
        if isinstance(obs_frame, dict): obs_frame = obs_frame.get('screen')
        if obs_frame is None: raise ValueError("Observation dictionary missing 'screen' key.")
        if not isinstance(obs_frame, (np.ndarray, torch.Tensor)): raise TypeError(f"Obs must be ndarray or tensor, got {type(obs_frame)}")
        if isinstance(obs_frame, np.ndarray):
            obs_tensor = torch.from_numpy(obs_frame).to(self.device).float()
            if obs_tensor.shape[-1] == 3 and obs_tensor.dim() == 3: obs_tensor = obs_tensor.permute(2, 0, 1)
        else:
            obs_tensor = obs_frame.to(self.device).float()
            if obs_tensor.shape[-1] == 3 and obs_tensor.dim() == 3: obs_tensor = obs_tensor.permute(2, 0, 1)
        if obs_tensor.dim() != 3 or obs_tensor.shape[0] != 3: raise ValueError(f"Expected CHW obs, got {obs_tensor.shape}")
        obs_tensor = obs_tensor / 255.0; obs_tensor = TF.resize(obs_tensor, list(self.img_size), antialias=True)
        return torch.clamp(obs_tensor, 0.0, 1.0)
    def forward(self, obs):
        processed_frame = self._preprocess(obs); self.buffer.append(processed_frame)
        seq = torch.stack(list(self.buffer), dim=0)
        features = self.vit(seq)
        if features.dim() > 1:
            if features.shape[0] == 1: features = features.squeeze(0)
            else: features = features.flatten()
        if features.dim() == 0: features = features.unsqueeze(0)
        if features.dim() != 1: raise RuntimeError(f"ViTWrapper expected 1D features, got {features.shape}")
        return features

# =============================================================================
# Checkpoint helper - Updated to save/load need modules and norm state
# =============================================================================
def save_checkpoint(path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, timesteps, rewards, norm_state):
    state = {'agent_state_dict': agent.state_dict(), 'vit_encoder_state_dict': vit_encoder.state_dict(),
             'rnd_model_state_dict': rnd_model.state_dict(), 'need_module_state_dict': need_module.state_dict(),
             'latent_bank_state_dict': latent_bank.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'rnd_optimizer_state_dict': rnd_optimizer.state_dict(), 'timesteps': timesteps, 'rewards': rewards,
             'running_norm_state': {'mean': norm_state.mean, 'var': norm_state.var, 'count': norm_state.count}}
    try: torch.save(state, path)
    except Exception as e: print(f"Error saving checkpoint to {path}: {e}")

def load_checkpoint(path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, running_norm):
    if not os.path.exists(path): print(f"Checkpoint file not found: {path}. Starting fresh."); return 0, []
    try:
        checkpoint = torch.load(path, map_location='cpu')
        agent.load_state_dict(checkpoint['agent_state_dict']); vit_encoder.load_state_dict(checkpoint['vit_encoder_state_dict'])
        rnd_model.load_state_dict(checkpoint['rnd_model_state_dict'])
        if 'need_module_state_dict' in checkpoint: need_module.load_state_dict(checkpoint['need_module_state_dict'])
        if 'latent_bank_state_dict' in checkpoint: latent_bank.load_state_dict(checkpoint['latent_bank_state_dict'])
        device = next(agent.parameters()).device
        agent.to(device); vit_encoder.to(device); rnd_model.to(device); need_module.to(device); latent_bank.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']); rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        timesteps = checkpoint.get('timesteps', 0); rewards = checkpoint.get('rewards', [])
        norm_state = checkpoint.get('running_norm_state')
        if norm_state: running_norm.mean = norm_state['mean']; running_norm.var = norm_state['var']; running_norm.count = norm_state['count']; print("Loaded running mean state.")
        print(f"Checkpoint loaded successfully from {path} (Timesteps: {timesteps})"); return timesteps, rewards
    except Exception as e: print(f"Error loading checkpoint: {e}. Starting from scratch."); return 0, []

# =============================================================================
# Training loop - Modified to apply key fixes, removed TensorBoard
# =============================================================================
def train(env_id: str = "VizdoomMyWayHome-v0",
          total_timesteps: int = 500_000,
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
          max_grad_norm: float = 0.5,
          save_dir: str = "models",
          vit_frame_history: int = 32,
          warmup_beta: int = 20000,
          final_beta_value: float = 0.05,
          freeze_vit_steps: int = 100000,
          load_checkpoint_path: str = None,
          save_freq: int = 50000
         ):

    # --- Initialization ---
    script_start_time = time.time() # Add script start time
    run_name = f"{env_id.split('-')[0]}__{os.path.basename(__file__)}__{int(time.time())}"
    os.makedirs(save_dir, exist_ok=True)
    # Log directory creation removed
    best_model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_ivrl_vit_best.pt")
    latest_model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_ivrl_vit_latest.pt")

    # TensorBoard writer removed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try: env = gym.make(env_id, render_mode=None)
    except Exception as e: raise RuntimeError(f"Failed to create env '{env_id}': {e}")
    print(f"Environment: {env_id} | Obs: {env.observation_space} | Act: {env.action_space}")

    # --- Model Initialization ---
    vit_encoder = ViTFeatureWrapper(frame_history=vit_frame_history, device=device, vit_model_class=InfiniViT).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)
    rnd_model = RNDModel(input_dim=512).to(device)
    latent_bank = LatentNeedBank(input_dim=512).to(device)
    need_module = NeedRewardModule(input_dim=512).to(device)
    running_norm = RunningMeanStd()

    # --- Optimizer Setup ---
    params_to_optimize = itertools.chain(agent.parameters(), need_module.parameters(), latent_bank.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=rnd_lr)
    if freeze_vit_steps > 0:
         print(f"Freezing ViT for first {freeze_vit_steps} steps.")
         for p in vit_encoder.parameters(): p.requires_grad = False
         params_to_optimize = itertools.chain(agent.policy.parameters(), agent.value_ext.parameters(), agent.value_int.parameters(),
                                          need_module.parameters(), latent_bank.parameters())
         optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # --- Checkpoint Loading ---
    start_timestep = 0; episode_rewards_history = []
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        start_timestep, episode_rewards_history = load_checkpoint(load_checkpoint_path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, running_norm)
        print(f"Resuming training from timestep {start_timestep}")
    elif load_checkpoint_path: print(f"Warning: Checkpoint path provided but not found: {load_checkpoint_path}")

    # --- Training Setup ---
    buffer = RolloutBuffer(); obs_data, info = env.reset(); vit_encoder.reset_buffer(); latent_bank.reset_state()
    episode_ext_rews_deque = deque(maxlen=100); episode_lengths_deque = deque(maxlen=100)
    current_episode_ext_rew = 0.0; current_episode_len = 0; total_episodes = 0
    best_mean_ext_reward = -float('inf') if not episode_rewards_history else max(episode_rewards_history, default=-float('inf'))
    num_iterations = total_timesteps // rollout_len
    print(f"Starting training for {total_timesteps} timesteps...")

    # --- Main Training Loop ---
    try:
        current_global_step = start_timestep # Initialize correctly for loaded checkpoints
        for iteration in range(start_timestep // rollout_len, num_iterations):
            start_time = time.time()
            # Calculate global step at the beginning of the iteration for consistency
            iteration_start_step = iteration * rollout_len

            # --- ViT Unfreezing ---
            # Check based on step count *before* starting rollout
            if freeze_vit_steps > 0 and iteration_start_step >= freeze_vit_steps:
                is_vit_frozen = any(not p.requires_grad for p in vit_encoder.parameters())
                if is_vit_frozen:
                    print(f"Unfreezing ViT encoder at step {iteration_start_step}")
                    for p in vit_encoder.parameters(): p.requires_grad = True
                    print("Recreating optimizer to include ViT parameters...")
                    params_to_optimize = itertools.chain(agent.parameters(), need_module.parameters(), latent_bank.parameters())
                    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
                    freeze_vit_steps = -1 # Prevent re-running

            # --- Rollout Phase ---
            agent.eval(); latent_bank.train(); rnd_model.eval(); need_module.eval()
            buffer.clear(); raw_intrinsic_rewards_this_rollout = []
            current_obs = obs_data # Use obs from previous iteration/reset

            for t in range(rollout_len):
                # Calculate exact global step for this transition
                global_step = iteration_start_step + t

                with torch.no_grad():
                    # Pass raw obs (dict/ndarray) to agent.get_action
                    action_t, logp_t, _, v_ext_t, v_int_t, features_t = agent.get_action(current_obs)

                action = action_t.cpu().item()
                next_obs_data, ext_rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                latent_bank.train(); rnd_model.eval(); need_module.eval() # Ensure modes
                with torch.no_grad():
                    rnd_novelty_t = rnd_model(features_t).squeeze(-1) # features_t is [1, FeatDim] -> rnd_novelty_t is [1]
                    other_need_vals = latent_bank(features_t, action_t)
                    all_need_vals = {**other_need_vals, "novelty": rnd_novelty_t}
                    int_rew_t, _ = need_module(features_t, all_need_vals) # Output [1]
                    raw_int_rew = int_rew_t.cpu().item()
                    raw_intrinsic_rewards_this_rollout.append(raw_int_rew)

                processed_frame_cpu = vit_encoder.buffer[-1].cpu()
                buffer.store(processed_frame_cpu, action, ext_rew, raw_int_rew, done,
                             logp_t.cpu().item(), v_ext_t.cpu().item(), v_int_t.cpu().item(),
                             features_t.squeeze(0).cpu())

                current_obs = next_obs_data; current_episode_ext_rew += ext_rew; current_episode_len += 1
                # Update current_global_step *after* processing step
                current_global_step = global_step + 1


                if done:
                    episode_ext_rews_deque.append(current_episode_ext_rew); episode_lengths_deque.append(current_episode_len)
                    total_episodes += 1
                    # Reset logic
                    current_obs, info = env.reset(); vit_encoder.reset_buffer(); latent_bank.reset_state()
                    current_episode_ext_rew = 0.0; current_episode_len = 0

            # --- Post-Rollout Processing ---
            if raw_intrinsic_rewards_this_rollout: running_norm.update(np.array(raw_intrinsic_rewards_this_rollout))
            mean_int_rew_rollout = np.mean(raw_intrinsic_rewards_this_rollout) if raw_intrinsic_rewards_this_rollout else float('nan') # Calculate mean intrinsic reward
            last_v_ext = last_v_int = 0.0
            if not done:
                 agent.eval()
                 with torch.no_grad():
                      _, last_v_ext_t, last_v_int_t, _ = agent.forward(obs=current_obs)
                      last_v_ext = last_v_ext_t.cpu().item(); last_v_int = last_v_int_t.cpu().item()
            buffer.finish_path(running_norm, last_v_ext, last_v_int, gamma, lam)

            # --- PPO Update Phase ---
            agent.train(); rnd_model.train(); latent_bank.train(); need_module.train()
            beta = beta_schedule(current_global_step, total_timesteps, warmup_beta, final_beta_value)

            # Store last losses for basic console logging
            last_policy_loss, last_value_loss, last_entropy_loss, last_rnd_loss = 0,0,0,0

            for epoch in range(K_epochs):
                for (obs_b, act_b, logp_b, adv_ext_b, adv_int_b,
                     ret_ext_b, ret_int_b, feat_b) in buffer.get_batches(batch_size, device):
                    logits, v_ext_pred, v_int_pred, _ = agent.forward(obs=None, features=feat_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    combined_adv = adv_ext_b + beta * adv_int_b
                    combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)
                    new_logp = dist.log_prob(act_b); ratio = torch.exp(new_logp - logp_b)
                    policy_loss_1 = -combined_adv * ratio; policy_loss_2 = -combined_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                    value_loss_ext = F.mse_loss(v_ext_pred, ret_ext_b); value_loss_int = F.mse_loss(v_int_pred, ret_int_b)
                    value_loss = value_loss_ext + value_loss_int; entropy_loss = -dist.entropy().mean()
                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_grad_norm); optimizer.step()
                    rnd_loss = rnd_model(feat_b.detach()).mean()
                    rnd_optimizer.zero_grad(); rnd_loss.backward(); torch.nn.utils.clip_grad_norm_(rnd_model.predictor.parameters(), max_grad_norm); rnd_optimizer.step()
                    # Store last batch losses
                    last_policy_loss = policy_loss.item(); last_value_loss = value_loss.item(); last_entropy_loss = entropy_loss.item(); last_rnd_loss = rnd_loss.item()


            # --- Logging and Saving ---
            end_time = time.time(); fps = int(buffer._size / (end_time - start_time))
            time_elapsed = int(end_time - script_start_time)
            mean_ext_rew = np.mean(episode_ext_rews_deque) if episode_ext_rews_deque else float('nan')
            mean_ep_len = np.mean(episode_lengths_deque) if episode_lengths_deque else float('nan')
            current_lr = optimizer.param_groups[0]['lr']

            # Replace the old print statement with SB3-style logging
            print(f"---------------------------------")
            print(f"| rollout/                |       |")
            print(f"|    ep_len_mean          | {mean_ep_len:<5.1f} |")
            print(f"|    ep_rew_mean          | {mean_ext_rew:<5.4f} |") # Updated format
            print(f"|    ep_int_rew_mean      | {mean_int_rew_rollout:<5.3f} |") # Added intrinsic reward
            print(f"|    exploration_beta     | {beta:<5.3f} |")
            print(f"|-------------------------|-------|")
            print(f"| time/                   |       |")
            print(f"|    fps                  | {fps:<5} |")
            print(f"|    iteration            | {iteration + 1:<5} |")
            print(f"|    time_elapsed         | {time_elapsed:<5} |")
            print(f"|    total_timesteps      | {current_global_step:<5} |")
            print(f"|-------------------------|-------|")
            print(f"| train/                  |       |")
            print(f"|    entropy_loss         | {last_entropy_loss:<5.3f} |")
            print(f"|    learning_rate        | {current_lr:<5.5f} |")
            print(f"|    policy_loss          | {last_policy_loss:<5.3f} |")
            print(f"|    rnd_loss             | {last_rnd_loss:<5.3f} |")
            print(f"|    value_loss           | {last_value_loss:<5.3f} |")
            print(f"---------------------------------")


            # Save best model
            if episode_ext_rews_deque and mean_ext_rew > best_mean_ext_reward :
                print(f"New best model! Mean reward: {mean_ext_rew:.3f}. Saving to {best_model_path}")
                best_mean_ext_reward = mean_ext_rew; episode_rewards_history.append(best_mean_ext_reward)
                save_checkpoint(best_model_path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, current_global_step, episode_rewards_history, running_norm)

            # Save latest checkpoint periodically
            # if (current_global_step // save_freq) > (iteration_start_step // save_freq):
            #      print(f"Saving latest checkpoint at step {current_global_step} to {latest_model_path}")
            #      save_checkpoint(latest_model_path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, current_global_step, episode_rewards_history, running_norm)

    except KeyboardInterrupt: print("\nTraining interrupted.")
    except Exception as e: print("\nError during training loop:"); import traceback; traceback.print_exc()
    finally:
        env.close(); print("\nSaving final model...")
        final_model_path = os.path.join(save_dir, f"{env_id.split('-')[0]}_ppo_ivrl_vit_final.pt")
        agent.to('cpu'); vit_encoder.to('cpu'); rnd_model.to('cpu'); need_module.to('cpu'); latent_bank.to('cpu')
        save_checkpoint(final_model_path, agent, vit_encoder, rnd_model, need_module, latent_bank, optimizer, rnd_optimizer, current_global_step if 'current_global_step' in locals() else start_timestep, episode_rewards_history, running_norm)
        print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO IVRL Minimal Changes")
    parser.add_argument("--env_id", type=str, default="VizdoomMyWayHome-v0")
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
    parser.add_argument("--rnd_lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="models")
    # log_dir argument removed
    parser.add_argument("--vit_frame_history", type=int, default=32)
    parser.add_argument("--warmup_beta", type=int, default=20000)
    parser.add_argument("--final_beta_value", type=float, default=0.05)
    parser.add_argument("--freeze_vit_steps", type=int, default=100000)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=50000)
    args = parser.parse_args()

    train(env_id=args.env_id, total_timesteps=args.total_timesteps, rollout_len=args.rollout_len,
          batch_size=args.batch_size, K_epochs=args.K_epochs, gamma=args.gamma, lam=args.lam,
          clip_range=args.clip_range, vf_coef=args.vf_coef, ent_coef=args.ent_coef, lr=args.lr,
          rnd_lr=args.rnd_lr, max_grad_norm=args.max_grad_norm, save_dir=args.save_dir,
          # log_dir argument removed
          vit_frame_history=args.vit_frame_history, warmup_beta=args.warmup_beta,
          final_beta_value=args.final_beta_value, freeze_vit_steps=args.freeze_vit_steps,
          load_checkpoint_path=args.load_checkpoint, save_freq=args.save_freq)