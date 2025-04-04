#!/bin/bash
#SBATCH --job-name=BasicViT_Policy_MyWayHome             # Job name
#SBATCH --partition=gpu-a100-40g      # Partition
#SBATCH --mail-type=ALL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nobr3541@colorado.edu     # Where to send mail
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --time=2-00:00:00                 # Time limit hrs:min:sec
#SBATCH --output=run_policy_basic.log           # Standard output and error log
#SBATCH --gres=gpu:1                      # request 1 gpu


# Execute the training script
python ppo_basic.py