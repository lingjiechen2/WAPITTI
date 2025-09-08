#!/bin/bash
# filepath: /mnt/petrelfs/fanyuyu/fyy/WAPITI-Code-Base-master/scripts/train/train_llama_logit_distill.sh

# Define model and watermark lists
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B"
    # "meta-llama/Llama-2-7b-hf"
    # "Qwen/Qwen2.5-3B"
)

WATERMARKS=(
    "kgw-k0-gamma0.25-delta1"
    "kgw-k0-gamma0.25-delta2"
    "kgw-k1-gamma0.25-delta1"
    "kgw-k1-gamma0.25-delta2"
    "kgw-k2-gamma0.25-delta1"
    "kgw-k2-gamma0.25-delta2"
)

# Output directory configuration
out_dir="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/watermark/"

# Function to check if all 5 checkpoints exist
check_complete() {
    local model="$1"
    local watermark="$2"
    local model_name="${model}-logit-watermark-distill-${watermark}"
    local target_path="${out_dir}${model_name}"
    echo $target_path

    # Check if all 5 checkpoint directories exist
    for i in 1000 2000 3000 4000 5000; do
        if [ ! -d "${target_path}/checkpoint-${i}" ]; then
            return 1  # Not complete
        fi
    done
    return 0  # Complete
}

# Create logs directory
mkdir -p logs

echo "Batch Training Job Submission"
echo "============================="

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for watermark in "${WATERMARKS[@]}"; do
        echo "Checking: ${model}-logit-watermark-distill-${watermark}"
        
        if check_complete "$model" "$watermark"; then
            echo "  SKIPPED - All checkpoints exist"
        else
            echo "  SUBMITTING..."
            echo "  Model: $model"
            echo "  Watermark: $watermark"
            
            sbatch scripts/train/train_logit_distill.sbatch "$watermark" "$model"
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Job submitted successfully!"
            else
                echo "  ✗ Failed to submit job!"
            fi
            sleep 1  # Small delay between submissions
        fi
        echo ""
    done
done

echo "All jobs processed!"
echo "Monitor with: squeue -u $(whoami)"