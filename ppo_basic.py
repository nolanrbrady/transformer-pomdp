#!/usr/bin/env python
# coding: utf-8

# In[5]:


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO
import torch as th
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import ResizeObservation
from vizdoom import gymnasium_wrapper

# Import model
from models.basic_vit import BasicViT


# In[16]:


# Custom Features Extractor using BasicViT
class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Print raw observation space for debugging
        print(f"Observation space: {observation_space}")
        
        screen_space = observation_space['screen']
        shape = screen_space.shape
            
        print(f"Shape from observation space: {shape}")
        
        # VecTransposeImage wrapper changes format to channels-first (C, H, W)
        if len(shape) == 3 and shape[0] == 3:
            # Channels first format (C, H, W)
            c, h, w = shape
        else:
            # Assume channels last format (H, W, C)
            h, w, c = shape

        h = 84
        w = 84
        c = 3
            
        print(f"Dimensions used: h={h}, w={w}, c={c}")
        
        # Calculate patches
        patch_size = 14
        n_h = h // patch_size
        n_w = w // patch_size
        n_patches = n_h * n_w
        
        print(f"Will create {n_patches} patches ({n_h}x{n_w}) with patch_size={patch_size}")

        # Set device
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Create the ViT with fixed in_channels=3 for RGB
        self.vit = BasicViT(
            img_size=(h, w),
            patch_size=patch_size,
            in_channels=3,  # IMPORTANT: Force to 3 for RGB images
            num_classes=features_dim,
            embed_dim=features_dim,
            num_blocks=6,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1,
            pad_if_needed=True,
            device=device,  # Explicitly pass the device
        )

    def forward(self, observations):
        # Handle dict observations
        if isinstance(observations, dict):
            obs = observations['screen']
        else:
            obs = observations
        
        # Ensure BCHW format for PyTorch
        obs = obs.float()
        if len(obs.shape) == 4:
            # If batch dimension exists
            if obs.shape[1] == 3:
                # Already in BCHW format
                pass
            elif obs.shape[3] == 3:
                # Convert BHWC to BCHW
                obs = obs.permute(0, 3, 1, 2)

        # Debug the model outputs
        features = self.vit(obs)
        # print(f"[ViT] Feature vector mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
        # print(f"[ViT] Feature sample: {features[0][:5]}")
        feature_var_across_batch = features.var(dim=0).mean().item()
        # print(f"[ViT] Avg feature variance across batch: {feature_var_across_batch:.6f}")
        return features

# # Custom Policy
# class CustomViTPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             *args,
#             **kwargs,
#             features_extractor_class=ViTFeatureExtractor,
#             features_extractor_kwargs=dict(features_dim=512),
#         )

print(f"PyTorch device check: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")

# Environment Setup
env = make_vec_env("VizdoomBasic-v0", n_envs=4)
obs_space = env.observation_space['screen']
act_space = env.action_space.n
img_height, img_width, channels = obs_space.shape

# Train PPO Agent
model = PPO(
    "MultiInputPolicy", 
    env,
    policy_kwargs=dict(
        features_extractor_class=ViTFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512)
    ),
    verbose=1
)
model.learn(total_timesteps=100_000)
model.save("ppo_basic_vit_vizdoom")


# In[15]:


from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ScreenOnlyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # Extract the screen space
        super().__init__(observation_space, features_dim)

        # This assumes the screen is (C, H, W)
        self.cnn = NatureCNN(observation_space.spaces["screen"], features_dim)

    def forward(self, observations):
        return self.cnn(observations["screen"])

# Setup environment
env = make_vec_env("VizdoomBasic-v0", n_envs=4)

# Set policy_kwargs to use custom feature extractor
policy_kwargs = dict(
    features_extractor_class=ScreenOnlyCNN,
    features_extractor_kwargs=dict(features_dim=512)
)

# Train the model
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100_000)


# In[ ]:




