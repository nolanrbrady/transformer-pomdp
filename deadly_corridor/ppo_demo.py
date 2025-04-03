import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import wandb
from wandb.integration.sb3 import WandbCallback

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from collections import deque

num_envs = 4
env_name = "humanoid-bench:h1hand-bookshelf_simple-v0"
max_steps = 20000000
learning_rate = 3e-5

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=num_envs, type=int)
parser.add_argument("--learning_rate", default=learning_rate, type=float)
parser.add_argument("--max_steps", default=max_steps, type=int)
parser.add_argument("--wandb_entity", default="robot-learning", type=str)
ARGS = parser.parse_args()


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed

    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Specify the render_mode so that the environment returns RGB frames.
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        env.action_space.seed(42 + rank)
        return env
    return _init


class EvalCallback(BaseCallback):
    def __init__(self, eval_every: int = 100000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_env(1)])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()
        return True

    def record_video(self) -> None:
        print("recording video")
        video = []
        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render().transpose(2, 0, 1)
            video.append(pixels)
        video = np.stack(video)
        wandb.log({"results/video": wandb.Video(video, fps=100, format="gif")})


def main(argv):
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    run = wandb.init(
        entity="robot-learning",
        project="humanoid-bench",
        name=f"ppo_{env_name}",
        monitor_gym=True,       # auto-upload videos of agent play
        save_code=True,         # optional
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=float(learning_rate),
        batch_size=512
    )
    model.learn(
        total_timesteps=max_steps,
        log_interval=1,
        callback=[
            WandbCallback(model_save_path=f"models/{run.id}", verbose=2),
            EvalCallback(),
        ]
    )
    model.save("ppo")
    print("Training finished")


if __name__ == '__main__':
    main(None)