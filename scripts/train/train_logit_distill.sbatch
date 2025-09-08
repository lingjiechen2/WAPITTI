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

GPU_COUNT=8
set -euo pipefail
source scripts/watermark_vector/utils.sh

# --- Positional args (sbatch passes them through) ---
# Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]
watermark=${1:?Usage: sbatch train_logit_distill.sbatch <watermark> <out_dir> <port> [llama_model]}
out_dir="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/watermark/"
port=$((20000 + RANDOM % 20000))
# model="meta-llama/Llama-2-7b-hf"
model=${2:-"meta-llama/Meta-Llama-3.1-8B"}

model_path="/mnt/lustrenew/mllm_safety-shared/models/huggingface"
dataset_path="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"

if [[ "${model%%/*}" == "meta-llama" ]]; then
    FSDP_LAYER="LlamaDecoderLayer"
elif [[ "${model%%/*}" == "Qwen" ]]; then
    FSDP_LAYER="Qwen2DecoderLayer"
fi

echo $FSDP_LAYER
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR
echo $MASTER_ADDR

# --- Paths / env (adjust PROJECT_ROOT if needed) ---
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/data"
cd "$PROJECT_ROOT"

model_name="${model}-logit-watermark-distill-${watermark}"
echo "${out_dir}${model_name}"

# --- Watermark arg switch ---
watermark_args=$(get_watermark_args "$watermark")
echo ${watermark_args}


# --- Batch & block sizes ---
if [[ "$watermark" == kth* ]]; then
    batch_size=32
    block_size=256
else
    batch_size=1
    block_size=128
fi

echo "[INFO] PROJECT_ROOT: $PROJECT_ROOT"
echo "[INFO] Watermark: $watermark"
echo "[INFO] Model (teacher): $model"
echo "[INFO] Output dir: ${out_dir}${model_name}"
echo "[INFO] GPUs: ${GPU_COUNT}  |  Master port: $port"

# Optional: tune NCCL for single-node multi-GPU stability
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

# Launch with srun so Slurm handles cgroups correctly
PYTHONPATH=/mnt/petrelfs/fanyuyu/lingjie_tmp/WAPITI-Code-Base-master \
srun --ntasks=1 torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${GPU_COUNT} \
  --master_port="${port}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$port \
  train_logit_distill.py \
    --model_name_or_path "${model_path}/${model}" \
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
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_LAYER}"