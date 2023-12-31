#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=05:00:00

export CUDA_VISIBLE_DEVICES=0 

for n_iter in 5 10 20 40 80; do
for k in 2 3 4 5; do

    for model in models/bert/best_final/*; do
        model_name=$(basename $model)

        out="./out/inference.random.${n_iter}.${k}.final.${model_name}.csv"

        if [[ ! -f $out ]]; then
            python3 inference_random.py \
                --from_adapters "${model}/final" \
                --training_set "./data/_nt_test.csv" \
                --n_mutations 3 \
                --k $k \
                --n_iter $n_iter \
                --local_batch_size 128 \
                --output $out
        fi
    done

    for model in models/bert/best_validation/*; do
        model_name=$(basename $model)

        out="./out/inference.random.${n_iter}.${k}.validation.${model_name}.csv"

        if [[ ! -f $out ]]; then

        python3 inference_random.py \
            --from_adapters "${model}/best" \
            --training_set "./data/_nt_test.csv" \
            --n_mutations 3 \
            --k $k \
            --n_iter $n_iter \
            --local_batch_size 128 \
            --output $out
        fi
    done

done
done
