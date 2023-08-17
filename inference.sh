#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

export CUDA_VISIBLE_DEVICES=0 

for model in models/bert/best_final/*; do
    model_name=$(basename $model)

    srun python3 inference.py \
        --from_adapters "${model}/final" \
        --from_mutator "./models/Mutator.final/${model_name}/final/model.bin" \
        --training_set "./data/_nt_test.csv" \
        --n_mutations 5 \
        --k 3 \
        --local_batch_size 128 \
        --output "inference.final.${model_name}.csv"

done

for model in models/bert/best_validation/*; do
    model_name=$(basename $model)

    srun python3 inference.py \
        --from_adapters "${model}/best" \
        --from_mutator "./models/Mutator.validation/${model_name}/final/model.bin" \
        --training_set "./data/_nt_test.csv" \
        --n_mutations 5 \
        --k 3 \
        --local_batch_size 128 \
        --output "inference.validation.${model_name}.csv"

done