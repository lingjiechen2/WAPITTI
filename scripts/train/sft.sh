dataset="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface/allenai/multi_lexsum"
# model="/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
model="/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen3-4B"

export PYTHONPATH=$HOME/trl:$PYTHONPATH
srun -p mllm_safety --gres=gpu:1 --quotatype=spot --cpus-per-task=8 --mem-per-cpu=8G --time=300 \
python experiments/sft.py \
    --model_name_or_path "${model}" \
    --dataset_name "${dataset}" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --num_proc 6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir /mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/finetuned/Qwen3-4B-legal \
    --bf16 True