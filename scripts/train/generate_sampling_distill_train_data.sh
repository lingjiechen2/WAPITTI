#!/bin/bash
model_path="/mnt/lustrenew/mllm_safety-shared/models/huggingface"
dataset_path="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
watermark=$1
llama=${2:-"meta-llama/Meta-Llama-3.1-8B"}
name=$3
min_new_tokens=256
max_new_tokens=256
num_samples=600

output_file="data/${watermark}_${name}_len${min_new_tokens}_${num_samples}_samples_dict.json"
output_train_file="data/sampling-distill-train-data-${watermark}.json"
watermark_config_file="experiments/watermark-configs/${watermark}-config.json"

echo "Watermark used: $watermark"
echo "Llama model used: $llama"

PYTHONPATH=/mnt/petrelfs/fanyuyu/lingjie_tmp/WAPITI-Code-Base-master \
srun -p mllm_safety --gres=gpu:1 --quotatype=spot --cpus-per-task=12 --mem-per-cpu=8G python experiments/generate_sampling_distill_train_data.py \
    --model_name "${model_path}/${llama}" \
    --dataset_name "${dataset_path}/Skylion007/openwebtext" \
    --dataset_split train \
    --data_field "text" \
    --streaming \
    --output_file "${output_file}" \
    --output_train_file "${output_train_file}" \
    --num_samples ${num_samples} \
    --min_new_tokens ${min_new_tokens} \
    --max_new_tokens ${max_new_tokens} \
    --prompt_length 50 \
    --seed 42 \
    --watermark_config_file "${watermark_config_file}" \
    --save_interval 64000 \
    --fp16 \
    --dataloader_batch_size 10000 \
    --batch_size 128
