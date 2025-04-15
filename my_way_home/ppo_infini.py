from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import torchvision.transforms.functional as TF

from gymnasium.wrappers import ResizeObservation
from vizdoom import gymnasium_wrapper

# Import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.infini_vit import InfiniViT

# Custom Features Extractor
class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, frame_history=32):
        super().__init__(observation_space, features_dim)
        self.frame_history = frame_history
        self.observation_space = observation_space

        h, w, c = 84, 84, 3
        patch_size = 14
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.vit = InfiniViT(
            img_size=(h, w),
            patch_size=patch_size,
            in_channels=3,
            num_classes=features_dim,
            embed_dim=features_dim,
            num_heads=4,
            mlp_ratio=2.0,
            memory_size=256,
            window_size=32,
            dropout=0.1,
            pad_if_needed=True,
            device=device,
            num_spatial_blocks=3,
            num_temporal_blocks=3,
            update_interval=5,
        )
        self.frame_buffers = None
        self.initialized = False

    def reset_history(self, batch_size):
        empty_frame = th.zeros((3, 84, 84), dtype=th.float32, device=self.vit.device)
        self.frame_buffers = [[empty_frame.clone() for _ in range(self.frame_history)] for _ in range(batch_size)]
        self.initialized = True

    def update_frames(self, observations):
        obs_screen = observations['screen'] if isinstance(observations, dict) else observations
        obs_tensor = th.as_tensor(obs_screen, device=self.vit.device).float() / 255.0

        batch_size = obs_tensor.shape[0]
        if not self.initialized or self.frame_buffers is None or len(self.frame_buffers) != batch_size:
            self.reset_history(batch_size)

        resized_obs = TF.resize(obs_tensor, (84, 84), antialias=True)

        for i in range(batch_size):
            self.frame_buffers[i].pop(0)
            self.frame_buffers[i].append(resized_obs[i].clone())

        return observations

    def forward(self, observations):
        self.update_frames(observations)

        batch_size = len(self.frame_buffers)

        features = []
        for i in range(batch_size):
            frame_sequence = th.stack(self.frame_buffers[i], dim=0)
            feature = self.vit(frame_sequence)
            features.append(feature)

        return th.stack(features, dim=0)

# Custom Policy with Larger Value Network
class CustomViTPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None,
                 activation_fn=nn.Tanh, features_extractor_class=ViTFeatureExtractor,
                 features_extractor_kwargs=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(pi=[512, 256], vf=[1024, 512, 256])],
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )

# Device check
print(f"PyTorch device check: {th.device('cuda' if th.cuda.is_available() else 'cpu')}")

# Setup Environment
env = make_vec_env("VizdoomMyWayHome-v0", n_envs=4)

# Train PPO Agent
model = PPO(
    policy=CustomViTPolicy,
    env=env,
    verbose=1
)
model.learn(total_timesteps=280_000)
model.save("ppo_vit_infini_my_way_home_large_vf")