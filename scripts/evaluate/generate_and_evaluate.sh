#!/bin/bash
dataset="c4"
model_family="meta-llama"
llama="meta-llama/Meta-Llama-3.1-8B"
ppl_model="meta-llama/Meta-Llama-3.1-8B"
tokenizer="meta-llama/Meta-Llama-3.1-8B"
num_tokens=200
prompt_length=50
num_samples=100
MODEL_PATH_PREFIX="/mnt/lustrenew/mllm_safety-shared/models/huggingface"
DATASET_PATH_PREFIX="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
TRAINED_MODEL_PATH_PREFIX="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/watermark/"

if [ "$dataset" = "c4" ]; then
    dataset_args="--dataset_name ${DATASET_PATH_PREFIX}/allenai/c4 \
    --dataset_config_name default \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_name wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text"
elif [ "$dataset" = "arxiv" ]; then
    dataset_args="--dataset_name scientific_papers \
    --dataset_config_name arxiv \
    --dataset_split test \
    --data_field article"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi

MODEL_BASE_DIR="meta-llama/Meta-Llama-3.1-8B-logit-watermark-distill-kgw-k1-gamma0.25-delta1"
# Collect all checkpoint folders inside the base dir, sorted numerically
models=($(ls -d "$TRAINED_MODEL_PATH_PREFIX/$MODEL_BASE_DIR"/checkpoint-* | sort -V))
printf '%s\n' "${models[@]}"
base_name=$(basename "$MODEL_BASE_DIR")
output_dir="/mnt/petrelfs/fanyuyu/fyy/WAPITI-Code-Base-master/results/${model_family}/${base_name}"

for model in "${models[@]}"; do
    checkpoint_name=$(basename "$model")
    output_file="${output_dir}/${checkpoint_name}.jsonl"
    echo $model
    echo $output_file

    (    
        PYTHONPATH=. \
        srun -p mllm_safety --gres=gpu:1 --quotatype=spot --cpus-per-task=8 --mem-per-cpu=8G --time=3000 \
        python experiments/generate_samples.py \
            --model_names ${model} \
            ${dataset_args} \
            --streaming \
            --fp16 \
            --output_file "${output_file}" \
            --num_samples ${num_samples} \
            --min_new_tokens ${num_tokens} \
            --max_new_tokens ${num_tokens} \
            --prompt_length ${prompt_length} \
            --batch_size 32 \
            --seed 42
            # --overwrite_output_file

        PYTHONPATH=. \
        srun -p mllm_safety --gres=gpu:1 --quotatype=spot --cpus-per-task=8 --mem-per-cpu=8G --time=3000 \
        python experiments/compute_metrics.py \
            --input_file "${output_file}"  \
            --output_file "${output_file}" \
            --tokenizer_name "${MODEL_PATH_PREFIX}/${tokenizer}" \
            --watermark_tokenizer_name "${MODEL_PATH_PREFIX}/${llama}" \
            --truncate \
            --num_tokens ${num_tokens} \
            --ppl_model_name "${MODEL_PATH_PREFIX}/${ppl_model}" \
            --fp16 \
            --batch_size 16 \
            --kgw_device cpu\
            --metrics p_value rep ppl \
            --overwrite_output_file

        # KTH watermark detection takes a while (several hours) and only requires CPU,
        # you can comment this out and run separately if desired
        # python watermarks/kth/compute_kth_scores.py \
        #     --tokenizer_name "${llama}" \
        #     --input_file "${output_file}" \
        #     --output_file "${output_file}" \
        #     --num_samples ${num_samples} \
        #     --num_tokens ${num_tokens} \
        #     --gamma 0.0 \
        #     --ref_dist_file "results/${dataset}/kth_ref_distribution_llama_${dataset}.json" \

        python experiments/compute_auroc.py \
            --input_file "${output_file}" \
            --output_file "${output_file}" \
            --auroc_ref_dist_file "results/${dataset}/auroc_ref_distribution_llama_${dataset}.json"
            # --kth_ref_dist_file "results/${dataset}/kth_ref_distribution_llama_${dataset}.json"
            # --overwrite_output_file \
    )&
    sleep 1
    echo "Started background job for $checkpoint_name (PID: $!)"
done

# Wait for all background jobs to complete
wait
echo "All checkpoints processed."