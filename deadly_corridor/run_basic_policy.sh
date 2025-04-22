#!/bin/bash
#SBATCH --job-name=vit_base_policy             # Job name
#SBATCH --partition=gpu-a100-40g      # Partition
#SBATCH --mail-type=ALL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nobr3541@colorado.edu     # Where to send mail
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --time=2-00:00:00                 # Time limit hrs:min:sec
#SBATCH --output=run_policy_basic.log           # Standard output and error log
#SBATCH --gres=gpu:1                      # request 1 gpu

# Activate your conda environment
source ~/.bashrc
conda activate vizdoom

# SDL2 from source
export SDL2_DIR=$HOME/libs/sdl2
export LD_LIBRARY_PATH=$SDL2_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$SDL2_DIR/lib:$LIBRARY_PATH
export CPATH=$SDL2_DIR/include:$CPATH
export PKG_CONFIG_PATH=$SDL2_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$SDL2_DIR:$CMAKE_PREFIX_PATH

# Execute the training script
python ppo_basic.py