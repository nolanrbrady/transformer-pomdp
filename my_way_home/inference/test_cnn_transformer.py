import torch
import torch.nn as nn
import numpy as np
import os
import sys
import gymnasium as gym

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppo_cnn_transformer import PPOAgent, CNNFeatureWrapper

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Set device
    device = torch_device
    env_id = "VizdoomMyWayHome-v0"
    # Path to the trained model checkpoint
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'VizdoomMyWayHome_ppo_cnn_transformer_best.pt')

    # Create environment (no rendering during step; we'll display frames manually)
    env = gym.make(env_id, render_mode=None)

    # Initialize feature extractor and agent
    cnn_encoder = CNNFeatureWrapper().to(device)
    action_dim = env.action_space.n
    agent = PPOAgent(cnn_encoder, action_dim).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint.get("agent_state_dict", {}))
    if "cnn_encoder_state_dict" in checkpoint:
        cnn_encoder.load_state_dict(checkpoint["cnn_encoder_state_dict"])

    agent.eval()
    cnn_encoder.eval()

    # Reset environment and model buffer
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    while True:
        # Preprocess observation into tensor
        if isinstance(obs, dict):
            obs_frame = obs['screen']
        else:
            obs_frame = obs
        obs_tensor = torch.tensor(obs_frame, dtype=torch.float32, device=device)
        if obs_tensor.shape[-1] == 3 and obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.permute(2, 0, 1)
        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

        # Select action using the trained agent
        with torch.no_grad():
            action, log_prob, entropy, value, feat = agent.get_action(obs_tensor)

        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().item())
        done = terminated or truncated

        # Update episode stats
        episode_reward += reward
        episode_length += 1

        obs = next_obs

        # If episode ended, print stats and reset environment
        if done:
            episode_count += 1
            print(f"Episode {episode_count}: length={episode_length} reward={episode_reward:.3f}")
            episode_reward = 0.0
            episode_length = 0
            obs, info = env.reset()
            agent.feature_buffer.clear()

    env.close()


if __name__ == "__main__":
    main()

