#!/bin/bash


#model_path="./pythia-12b-4bit-bbq"
#model_path="EleutherAI/pythia-12b"
model_path="/home/bstahl/bbq/pythia-12b/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
tasks="hellaswag"
output_dir="model_eval_results"


mkdir -p "$output_dir"

echo "========== Starting =========="
echo "model: $model_path"
echo "saving results to: $output_dir"

lm_eval --model hf \
    --model_args pretrained=$model_path,trust_remote_code=true \
    --tasks $tasks \
    --device cuda:0 \
    --batch_size auto \
    --output_path $output_dir \
    --log_samples \

echo "========== Done =========="
