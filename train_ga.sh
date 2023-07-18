#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

export CUDA_VISIBLE_DEVICES=0 
srun python3 train_ga.py