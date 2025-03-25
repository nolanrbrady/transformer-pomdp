# Robotics Transformers: Final Project

This repository contains implementations of reinforcement learning agents for robotics using a modified Vision Transformer (TinyViT) as the policy network. We experiment with REINFORCE, Actor-Critic (AC), and Proximal Policy Optimization (PPO) algorithms by training agents in simulated environments (e.g., Vizdoom).

## Overview

The project focuses on leveraging a non-pretrained Tiny Vision Transformer model to build policies for RL agents. The main models are implemented in:

- **models/huggingface_vit_policy.py**: Contains the modified TinyViTPolicy and TinyViTActorCriticPolicy. 
- **models/basic_vit.py**: Provides the baseline TinyViT implementation used as the backbone.

The reinforcement learning algorithms are implemented in **basic_rl_agent.py**, which supports the following:

- **REINFORCE**: Actor-only policy gradient method.
- **Actor-Critic (AC)**: Uses a combined policy with an integrated value head.
- **PPO**: Implements Proximal Policy Optimization with a clipped surrogate objective, utilizing a combined actor-critic network.

## Dependencies

This project relies on the following libraries:

- Python 3.8+ (recommended: Python 3.11)
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (with Vizdoom gymnasium wrapper)
- [Vizdoom](https://github.com/mwydmuch/ViZDoom)
- NumPy

Install the dependencies via:

```bash
pip install -r requirements.txt
```

Ensure you also have Vizdoom and its gymnasium wrapper installed as needed.

## Environment Setup

This project uses simulated environments for training. For example, the **VizdoomBasic-v0** environment is used in the training scripts provided in `basic_rl_agent.py`.

Before running training, make sure your environment is set up to run Vizdoom-based simulations.

## Training

To train an agent, run the main script:

```bash
python basic_rl_agent.py
```

You will be prompted to choose an algorithm:

- **reinforce**: Trains an agent using the REINFORCE algorithm.
- **ac**: Trains an actor-critic agent using a combined network (TinyViTActorCriticPolicy).
- **ppo**: Trains an agent using Proximal Policy Optimization (utilizes the combined actor-critic model with proper evaluation support).

During training, metrics such as rewards, losses, and entropy are printed to the console. The training logs for rewards and losses are saved as numpy arrays (e.g., `basic_reinforce_total_rewards.npy`, `basic_ppo_total_losses.npy`).

## Model Architecture

- **TinyViTPolicy**: A lightweight Vision Transformer used as the policy network for REINFORCE.
- **TinyViTActorCriticPolicy**: Extends the TinyViTPolicy by integrating a value head, making it suitable for Actor-Critic and PPO algorithms. Recent modifications ensure that the network correctly handles pointer evaluations without returning ambiguous tuple outputs.

## Code Structure

- `basic_rl_agent.py`: Contains implementations for REINFORCE, Actor-Critic, and PPO agents along with the training loops.
- `models/huggingface_vit_policy.py`: Contains the modified ViT architectures for both pure policy and actor-critic setups.
- `models/basic_vit.py`: Provides the foundational TinyViT model implementation.

## Experimental Results

During training, the agent reports metrics like total rewards, policy loss, value loss, and entropy. The results are archived in .npy files for further analysis:

- **REINFORCE**: `basic_reinforce_total_rewards.npy`, `basic_reinforce_total_losses.npy`
- **Actor-Critic**: `basic_ac_total_rewards.npy`, `basic_ac_total_losses.npy`
- **PPO**: `basic_ppo_total_rewards.npy`, `basic_ppo_total_losses.npy`

## Troubleshooting & Notes

- If you encounter issues with value estimation in PPO, ensure that you are using the combined actor-critic model (i.e., pass `value_head=None` when initializing the agent).
- Input normalization is crucial: screen images should be normalized by dividing pixel values by 255.
- For further modifications or experiments, consider adjusting the hyperparameters in `basic_rl_agent.py`.

## Contributions

Contributions and improvements are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.