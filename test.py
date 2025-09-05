from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/mnt/lustrenew/mllm_safety-shared/models/huggingface/meta-llama/Llama-3.2-3B-Instruct/", trust_remote_code=True)

# Print the architecture
print(model)

# Or recursively print submodules
for name, module in model.named_modules():
    print(name, module.__class__.__name__)
