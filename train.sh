#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_name test \
    --model_version 2 \
    --learning_rate 0.0004 \
    --training_set "./data/_test.csv" \
    --validation_set "./data/_val.csv" \
    --max_length 512 \
    --n_epochs 1 \
    --global_batch_size 32 \
    --local_batch_size 16 \
    --num_workers 16 \
    --mask