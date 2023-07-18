#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

export CUDA_VISIBLE_DEVICES=0 
srun python3 train_ga.py \
    --model_name "Mutator" \
    --model_version "0.1" \
    --from_tokenizer "Rostlab/prot_bert_bfd" \
    --from_model "Rostlab/prot_bert_bfd" \
    --from_adapters "./models/hps/LR0.001_BS256_P0.05/best/" \
    --training_set "./data/non_thermo.csv" \
    --global_batch_size 128 \
    --local_batch_size 128 \
    --n_epochs 10 \
    --population_size 4 \
    --offspring_size 2 \