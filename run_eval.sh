#!/bin/bash


#model_path="./pythia-12b-4bit-bbq"
model_path="EleutherAI/pythia-12b"
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
