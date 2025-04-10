# Custom PPO with RND and ViT for ViZDoom Deadly Corridor (Full Version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import torchvision.transforms.functional as TF
from vizdoom import gymnasium_wrapper

# === Import your ViT Model ===
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
        feat = self.vit(obs)
        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value.squeeze(-1), feat

    def get_action(self, obs):
        logits, value, feat = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, feat.detach()


# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values, self.features = [], [], [], []
        self.advantages, self.returns = [], []

    def store(self, obs, action, reward, done, log_prob, value, feature):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
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
                torch.tensor([self.advantages[i] for i in indices[start:end]]),
                torch.tensor([self.returns[i] for i in indices[start:end]]),
                torch.stack([self.features[i] for i in indices[start:end]])
            ]

    def clear(self):
        self.__init__()


# === Preprocessing ViT Feature Extractor ===
class ViTFeatureWrapper(nn.Module):
    def __init__(self, frame_history=4):
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

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs / 255.0
        obs = TF.resize(obs, (84, 84), antialias=True)
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        for i in range(obs.shape[0]):
            self.buffer.append(obs[i])
        while len(self.buffer) < self.frame_history:
            self.buffer.append(obs[0])
        stacked = torch.stack(list(self.buffer), dim=0)
        return self.vit(stacked.unsqueeze(0)).squeeze(0)


# === Training Loop ===
def train(env_id="VizdoomMyWayHome-v0", total_timesteps=1_000_000, rollout_len=2048, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id, render_mode=None)
    vit_encoder = ViTFeatureWrapper().to(device)
    rnd_model = RNDModel(input_dim=512).to(device)
    agent = PPOAgent(vit_encoder, env.action_space.n).to(device)

    optimizer = torch.optim.Adam(list(agent.parameters()), lr=2.5e-4)
    rnd_optimizer = torch.optim.Adam(rnd_model.predictor.parameters(), lr=1e-4)

    buffer = RolloutBuffer()

    obs, _ = env.reset()
    for step in range(0, total_timesteps, rollout_len):
        buffer.clear()
        ep_rewards = []

        for t in range(rollout_len):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value, feature = agent.get_action(obs_tensor)
                intrinsic_reward = rnd_model(feature).cpu().item()

            next_obs, extrinsic_reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            total_reward = extrinsic_reward + 0.01 * intrinsic_reward

            buffer.store(obs_tensor.squeeze(0), action.item(), total_reward, done, log_prob.item(), value.item(), feature.squeeze(0))
            obs = next_obs

            if done:
                obs, _ = env.reset()

        buffer.finish_path()

        # PPO Updates
        for obs_batch, act_batch, logp_batch, adv_batch, ret_batch, feat_batch in buffer.get_batches(batch_size):
            logits, values, _ = agent.forward(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(act_batch)

            ratio = torch.exp(new_log_probs - logp_batch)
            clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv_batch
            policy_loss = -torch.min(ratio * adv_batch, clip_adv).mean()

            value_loss = F.mse_loss(values, ret_batch)
            entropy_loss = dist.entropy().mean()

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # RND update
            rnd_loss = rnd_model(feat_batch).mean()
            rnd_optimizer.zero_grad()
            rnd_loss.backward()
            rnd_optimizer.step()

        print(f"Step {step}: Loss = {total_loss.item():.3f}, RND Loss = {rnd_loss.item():.3f}")

if __name__ == "__main__":
    train()
