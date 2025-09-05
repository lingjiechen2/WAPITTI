#!/bin/bash
#
#SBATCH -p mllm_safety
#SBATCH --quotatype=spot
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH -t 24:00:00
#SBATCH -J distill-train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- Positional args (sbatch passes them through) ---
# Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]
watermark=${1:?Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]}
out_dir=${2:?Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]}
port=${3:?Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]}
llama=${4:-"meta-llama/Llama-3.2-3B-Instruct"}
model_path="/mnt/lustrenew/mllm_safety-shared/models/huggingface"
dataset_path="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"

# --- Paths / env (adjust PROJECT_ROOT if needed) ---
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/data"
cd "$PROJECT_ROOT"

# # Make sure Python can import your local package
# export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

model_name="${llama}-logit-watermark-distill-${watermark}"
echo "${out_dir}${model_name}"

# --- Watermark arg switch ---
if [ "$watermark" = "aar-k2" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 2"
elif [ "$watermark" = "aar-k3" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 3"
elif [ "$watermark" = "aar-k4" ]; then
    watermark_args="--watermark_type aar --aar_watermark_k 4"
elif [ "$watermark" = "kgw-k0-gamma0.25-delta1" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 1.0 \
    --kgw_watermark_seeding_scheme simple_0"
elif [ "$watermark" = "kgw-k0-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_0"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta1" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 1.0 \
    --kgw_watermark_seeding_scheme simple_1"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_1"
elif [ "$watermark" = "kgw-k2-gamma0.25-delta2" ]; then
    watermark_args="--watermark_type kgw \
    --kgw_watermark_gamma 0.25 \
    --kgw_watermark_delta 2.0 \
    --kgw_watermark_seeding_scheme simple_2"
elif [ "$watermark" = "kth-shift1" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 1"
elif [ "$watermark" = "kth-shift2" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 2"
elif [ "$watermark" = "kth-shift4" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 4"
elif [ "$watermark" = "kth-shift256" ]; then
    watermark_args="--watermark_type kth \
    --kth_watermark_key_len 256 \
    --kth_watermark_num_shifts 256"
else
    echo "Unsupported watermark type ${watermark}."
    exit 1
fi

# --- Batch & block sizes ---
if [[ "$watermark" == kth* ]]; then
    batch_size=32
    block_size=256
else
    batch_size=8
    block_size=256
fi

echo "[INFO] PROJECT_ROOT: $PROJECT_ROOT"
echo "[INFO] Watermark: $watermark"
echo "[INFO] Model (teacher): $llama"
echo "[INFO] Output dir: ${out_dir}${model_name}"
echo "[INFO] GPUs: 8  |  Master port: $port"

# Optional: tune NCCL for single-node multi-GPU stability
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

# Launch with srun so Slurm handles cgroups correctly
PYTHONPATH=/mnt/petrelfs/fanyuyu/lingjie_tmp/WAPITI-Code-Base-master \
srun --ntasks=1 \
  torchrun --nnodes=1 --nproc_per_node=8 --master_port="${port}" train_logit_distill.py \
    --model_name_or_path "${model_path}/${llama}" \
    --dataset_name "${dataset_path}/Skylion007/openwebtext" \
    --streaming \
    --per_device_train_batch_size "${batch_size}" \
    --gradient_accumulation_steps 1 \
    --do_train \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size "${block_size}" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --fp16 True \
    --gradient_checkpointing True \
    ${watermark_args} \
    --watermark_seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
