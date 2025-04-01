#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO
import torch as th
import torch.nn as nn
from collections import deque
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import ResizeObservation
from vizdoom import gymnasium_wrapper
import vizdoom.gymnasium_wrapper.scenarios.basic
import vizdoom.gymnasium_wrapper.scenarios.corridor
import vizdoom.gymnasium_wrapper.scenarios.defend_the_center
import vizdoom.gymnasium_wrapper.scenarios.defend_the_line
import vizdoom.gymnasium_wrapper.scenarios.health_gathering
import vizdoom.gymnasium_wrapper.scenarios.my_way_home
import vizdoom.gymnasium_wrapper.scenarios.prediction
import vizdoom.gymnasium_wrapper.scenarios.take_cover
import vizdoom.gymnasium_wrapper
vizdoom.gymnasium_wrapper.register_all_scenarios()

# Import model
from models.infini_vit import InfiniViT

def make_headless_env(env_id="VizdoomCorridor-v0"):
    def _init():
        env = gym.make(env_id, render_mode=None)  # Ensure headless mode
        env.unwrapped.game.set_window_visible(False)  # Prevent GUI on HPC
        env.unwrapped.game.set_render_hud(False)
        env.unwrapped.game.set_render_decals(False)
        env.unwrapped.game.set_render_particles(False)
        env.unwrapped.game.set_render_effects_sprites(False)
        env.unwrapped.game.set_render_corpses(False)
        env.unwrapped.game.set_screen_format(gymnasium_wrapper.vizdoom.ScreenFormat.RGB24)
        return env
    return _init


# In[ ]:


# Custom Features Extractor using BasicViT
class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, frame_history=4):
        super().__init__(observation_space, features_dim)
        
        # Frame history length
        self.frame_history = frame_history
        self.observation_space = observation_space
        
        # Print raw observation space for debugging
        print(f"Observation space: {self.observation_space}")
        
        screen_space = self.observation_space['screen']
        shape = screen_space.shape
            
        print(f"Shape from observation space: {shape}")
        
        # VecTransposeImage wrapper changes format to channels-first (C, H, W)
        if len(shape) == 3 and shape[0] == 3:
            # Channels first format (C, H, W)
            c, h, w = shape
        else:
            # Assume channels last format (H, W, C)
            h, w, c = shape

        # Set the desired image dimensions
        h = 84
        w = 84
        c = 3
            
        print(f"Extracted dimensions: h={h}, w={w}, c={c}")
        
        # Calculate patches
        patch_size = 14
        n_h = h // patch_size
        n_w = w // patch_size
        n_patches = n_h * n_w
        
        print(f"Will create {n_patches} patches ({n_h}x{n_w}) with patch_size={patch_size}")

        # Set device
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Create the ViT with fixed in_channels=3 for RGB
        self.vit = InfiniViT(
            img_size=(h, w),
            patch_size=patch_size,
            in_channels=3,  # IMPORTANT: Force to 3 for RGB images
            num_classes=features_dim,
            embed_dim=features_dim,
            num_heads=8,
            mlp_ratio=2.0,
            memory_size=128,
            window_size=1,
            dropout=0.1,
            pad_if_needed=True,
            device=device,
            num_spatial_blocks=3,
            num_temporal_blocks=3,
        )
        
        # Initialize frame history buffers 
        self.frame_buffers = None
        self.initialized = False
        
    def reset_history(self, batch_size):
        """Initialize or reset frame history for all environments"""
        # Create a default empty tensor for initial frames
        empty_frame = th.zeros((3, *self.observation_space['screen'].shape[1:]), 
                               dtype=th.float32,
                               device=self.vit.device)
        
        # Create a frame buffer for each environment in the batch
        self.frame_buffers = []
        for _ in range(batch_size):
            buffer = []
            for _ in range(self.frame_history):
                # Create a new tensor for each frame to avoid reference issues
                buffer.append(empty_frame.clone())
            self.frame_buffers.append(buffer)
        
        self.initialized = True
        print(f"Initialized frame buffers for {batch_size} environments with {self.frame_history} frames each")

    def update_frames(self, observations):
        """Update frame history with new observations"""
        if isinstance(observations, dict):
            obs = observations['screen']
        else:
            obs = observations
            
        # Initialize buffers if needed
        batch_size = obs.shape[0]
        if not self.initialized or self.frame_buffers is None or len(self.frame_buffers) != batch_size:
            self.reset_history(batch_size)
        
        # Update each environment's frame buffer by replacing the oldest frame
        for i in range(batch_size):
            # Remove oldest frame and add new frame
            self.frame_buffers[i].pop(0)
            self.frame_buffers[i].append(obs[i].clone().to(self.vit.device))
            
        return obs

    def forward(self, observations):
        """Process observations through the temporal ViT model"""
        # Update frame history with new observations
        obs = self.update_frames(observations)
        
        # Create a batch of temporal sequences
        batch_size = obs.shape[0]
        features = []
        
        # Process each environment's frame history separately
        for i in range(batch_size):
            # Stack frames along first dimension to create temporal sequence (T, C, H, W)
            frame_sequence = th.stack(self.frame_buffers[i], dim=0)
            
            # Pass the sequence through TemporalViT
            feature = self.vit(frame_sequence)
            features.append(feature)
            
        # Stack all features into a batch
        features = th.stack(features, dim=0)
        return features

print(f"PyTorch device check: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")

# Environment Setup
env = make_vec_env(make_headless_env("VizdoomCorridor-v0"), n_envs=8)
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
model.learn(total_timesteps=2_000_000)
model.save("ppo_infini_vit_vizdoom")


# In[ ]:
