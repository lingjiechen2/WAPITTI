#!/bin/bash
# filepath: run_watermark_vectors.sh

source scripts/watermark_vector/utils.sh

# Configuration
BASE_OUTPUT_DIR="/mnt/petrelfs/fanyuyu/fyy/WAPITI-Code-Base-master/results/watermark_vectors"
NUM_SAMPLES=96
BATCH_SIZE=32
USER="fanyuyu"
MAX_SUBMITTED_JOBS=10

# Model definitions
VANILLA_MODEL="/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3.1-8B"
declare -a TESTED_MODELS=("${VANILLA_MODEL}")

# Define watermark models directory
WATERMARK_MODELS_DIR="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/watermark/meta-llama/Meta-Llama-3.1-8B-logit-watermark-distill-kgw-k0-gamma0.25-delta1"

# Find all subdirectories under WATERMARK_MODELS_DIR
echo "Finding watermark models in ${WATERMARK_MODELS_DIR}..."
WATERMARK_MODELS=($(find "${WATERMARK_MODELS_DIR}" -mindepth 1 -maxdepth 1 -type d))
echo "Found ${#WATERMARK_MODELS[@]} watermark models"

# Define coefficient range (adjust as needed)
declare -a COEFFICIENTS=($(seq 0 0.3 4))

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Function to check if results exist
check_results_exist() {
    local watermark_model=$1
    local coefficient=$2
    local tested_model=$3
    local base_output_dir=$4
    
    # Extract names once
    local watermark_dirname=$(basename "$(dirname "$watermark_model")")
    local watermark_name=$(basename "$watermark_model")
    local tested_name=$(basename "$tested_model")

    # Construct path
    local path_name="${tested_name}/${watermark_dirname}/${watermark_name}"
    local normalized_path="${path_name//-/_}"
    local expected_dir="${base_output_dir}/${normalized_path}/coefficient_${coefficient}"
    echo $expected_dir

    # Check if both config.json and scores.json exist
    if [[ -f "${expected_dir}/config.json" && -f "${expected_dir}/scores.json" ]]; then
        return 0  # Results exist
    else
        return 1  # Results don't exist
    fi
}



# Function to run single experiment
run_experiment() {
    local watermark_model=$1
    local coefficient=$2
    local tested_model=$3
    local vanilla_model=$4
    
    # Check if results already exist
    if check_results_exist "$watermark_model" "$coefficient" "$tested_model" "$BASE_OUTPUT_DIR"; then
        echo "Results already exist for $(basename "$watermark_model") coeff=$coefficient model=$(basename "$tested_model"), skipping..."
        return 0
    fi
    
    echo "Starting experiment: $(basename "$watermark_model") coeff=$coefficient model=$(basename "$tested_model")"
    
    PYTHONPATH=. \
    srun -p mllm_safety --gres=gpu:1 --quotatype=spot --cpus-per-task=8 --mem-per-cpu=8G --time=3000 \
    python experiments/watermark_vector.py \
        --watermark_base_dir $(basename "$WATERMARK_MODELS_DIR") \
        --watermark_model "$watermark_model" \
        --coefficient "$coefficient" \
        --vanilla_model_name "$vanilla_model" \
        --tested_model_name "$tested_model" \
        --base_output_dir "$BASE_OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE"
        # Remove --skip_if_exists flag
    
    echo "Completed experiment: $(basename "$watermark_model") coeff=$coefficient model=$(basename "$tested_model")"
}

echo "Starting watermark vector experiments..."
echo "Watermark models found: ${#WATERMARK_MODELS[@]}"
echo "Coefficients: ${#COEFFICIENTS[@]}"
echo "Tested models: ${#TESTED_MODELS[@]}"
echo "Total experiments: $((${#WATERMARK_MODELS[@]} * ${#COEFFICIENTS[@]} * ${#TESTED_MODELS[@]}))"

# Triple nested loop for all combinations
for tested_model in "${TESTED_MODELS[@]}"; do
    for watermark_model in "${WATERMARK_MODELS[@]}"; do
        for coefficient in "${COEFFICIENTS[@]}"; do
            (
                run_experiment "$watermark_model" "$coefficient" "$tested_model" "$VANILLA_MODEL"
            ) &
            echo "------------------------------------------------------------"
            sleep 0.5
            wait_for_jobs_below_threshold "${USER}" ${MAX_SUBMITTED_JOBS}
        done
    done
done

# Wait for all remaining background jobs to complete
echo "Waiting for all remaining jobs to complete..."
wait

echo "All watermark vector experiments completed!"
echo "Results saved to: $BASE_OUTPUT_DIR"

# Optional: Generate summary
echo "Generating experiment summary..."
find "$BASE_OUTPUT_DIR" -name "config.json" | wc -l | xargs echo "Total completed experiments:"
find "$BASE_OUTPUT_DIR" -name "scores.json" | wc -l | xargs echo "Experiments with scores computed:"