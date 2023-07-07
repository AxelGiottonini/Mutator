#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_name test \
    --model_version 0.0 \
    --learning_rate 0.0001 \
    --training_set "./data/_test.csv" \
    --validation_set "./data/_val.csv" \
    --max_length 400 \
    --n_epochs 5 \
    --batch_size 256 \
    --local_batch_size 16 \
    --num_workers 16 