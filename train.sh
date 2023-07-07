#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./src/train.py \
    --model_name test \
    --model_version 1.1 \
    --learning_rate 0.000 \
    --training_set "./data/train.50k.csv" \
    --validation_set "./data/val.500.csv" \
    --max_length 400 \
    --n_epochs 20 \
    --batch_size 256 \
    --local_batch_size 16 \
    --num_workers 16 