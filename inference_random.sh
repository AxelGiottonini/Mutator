#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

export CUDA_VISIBLE_DEVICES=0 
srun python3 inference_random.py \
    --model_name "inference" \
    --model_version "none" \
    --from_adapters "./models/bert/best_final/LR0.0004_BS256_P0.05/final/" \
    --training_set "./data/non_thermo.csv" \
    --n_mutations 3 \
    --k 3 \
    --n_iter 3\
    --local_batch_size 128 \