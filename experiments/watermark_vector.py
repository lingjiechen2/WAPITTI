import argparse
import os
import copy
import sys
sys.path.append(('../'))
sys.path.append(('../../'))
import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from task_vector import TaskVector

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType
import logging
from experiments.compute_metrics import compute_seq_rep_n, compute_total_rep_n, compute_ppl


def parse_args():
    parser = argparse.ArgumentParser(description="Atomic watermark vector computation")
    parser.add_argument("--watermark_base_dir", type=str, required=True, 
                       help="Name of the watermark model's parent dir, for saving architecture")
    parser.add_argument("--watermark_model", type=str, required=True, 
                       help="Name of the watermark model to process")
    parser.add_argument("--coefficient", type=float, required=True,
                       help="Coefficient value to process")
    parser.add_argument("--vanilla_model_name", type=str, required=True,
                       help="Name of the vanilla/base model")
    parser.add_argument("--tested_model_name", type=str, required=True,
                       help="Name of the model to apply watermark vector to")
    parser.add_argument("--base_output_dir", type=str, 
                       default="/remote-home/miintern1/watermark-learnability/results/watermark_vectors",
                       help="Base directory for storing results")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--skip_if_exists", action="store_true",
                       help="Skip processing if results already exist")
    return parser.parse_args()

def sanitize_path(name):
    """Convert model name to filesystem-safe path"""
    return name.replace("/", "_").replace("-", "_")

def create_output_structure(base_dir, tested_model_name, watermark_model_name, checkpoint_name, coefficient):
    """Create hierarchical directory structure and return paths"""
    tested_model_safe = sanitize_path(tested_model_name)
    watermark_safe = sanitize_path(watermark_model_name)
    checkpoint_safe = sanitize_path(checkpoint_name)
    coef_dir = f"coefficient_{coefficient:.1f}"
    
    output_dir = os.path.join(base_dir, tested_model_safe, watermark_safe, checkpoint_safe, coef_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        'config': os.path.join(output_dir, 'config.json'),
        'outputs': os.path.join(output_dir, 'outputs.json'),
        'scores': os.path.join(output_dir, 'scores.json')
    }
    
    return output_dir, paths

def check_completion(paths):
    """Check if all required files exist"""
    return all(os.path.exists(path) for path in paths.values())

def save_config(path, config_data):
    """Save configuration file"""
    with open(path, 'w') as f:
        json.dump(config_data, f, indent=4)

def save_outputs(path, outputs_data):
    """Save outputs file"""
    with open(path, 'w') as f:
        json.dump(outputs_data, f, indent=4)

def save_scores(path, scores_data):
    """Save scores file"""
    with open(path, 'w') as f:
        json.dump(scores_data, f, indent=4)

def setup_logging(output_dir):
    """Setup logging for this specific run"""
    log_file = os.path.join(output_dir, 'run.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite log file for each run
    )
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def compute_metrics_for_outputs(vanilla_outputs, watermarked_outputs, prompt_texts, tokenizer, ppl_model_name, batch_size=16, fp16=True):
    """Compute perplexity and repetition metrics for both vanilla and watermarked outputs"""
    # Prepare data in the format expected by compute_metrics functions
    samples_dict = {
        "vanilla": {
            "model_text": vanilla_outputs,
            "full_model_text": [prompt + " " + output for prompt, output in zip(prompt_texts, vanilla_outputs)],
            "prompt_text": prompt_texts
        },
        "watermarked": {
            "model_text": watermarked_outputs,
            "full_model_text": [prompt + " " + output for prompt, output in zip(prompt_texts, watermarked_outputs)],
            "prompt_text": prompt_texts
        }
    }

    # Compute repetition metrics
    logging.info("Computing repetition metrics...")
    for model_name, sd in samples_dict.items():
        samples = sd["model_text"]
        rep_3_results = compute_seq_rep_n(samples, tokenizer, n=3)
        total_rep_3_results = compute_total_rep_n(samples, tokenizer, n=3)
        sd.update(rep_3_results)
        sd.update(total_rep_3_results)
    
    # Compute perplexity metrics
    logging.info("Computing perplexity metrics...")
    compute_ppl(samples_dict, ppl_model_name, batch_size, fp16)
    
    return samples_dict


def main():
    args = parse_args()
    
    # Model configurations
    watermark_config_path = os.path.join(args.watermark_model, "watermark_config.json")
    if os.path.exists(watermark_config_path):
        watermark_config = json.load(open(watermark_config_path))
    else:
        raise ValueError(f"No watermark_config.json found under: {args.watermark_model}")
    
    vanilla_model_name = args.vanilla_model_name
    tested_model_name = args.tested_model_name
    
    # Create output structure
    watermark_checkpoint_name = os.path.basename(args.watermark_model)
    tested_model_path_name = os.path.basename(args.tested_model_name)
    output_dir, paths = create_output_structure(
        args.base_output_dir, tested_model_path_name, args.watermark_base_dir, watermark_checkpoint_name, args.coefficient
    )
    
    # Setup logging
    setup_logging(output_dir)
    
    # Check if already completed
    if args.skip_if_exists and check_completion(paths):
        logging.info(f"Results already exist for {watermark_checkpoint_name} coefficient {args.coefficient}, skipping...")
        return
    
    logging.info(f"Processing {watermark_checkpoint_name} with coefficient {args.coefficient}")
    logging.info(f"Output directory: {output_dir}")
    
    # Save configuration
    config_data = {
        "watermark_name": args.watermark_base_dir,
        "watermark_model": watermark_checkpoint_name,
        "tested_model_name": tested_model_name,
        "vanilla_model_name": vanilla_model_name,
        "coefficient": args.coefficient,
        "watermark_config": watermark_config,
        "generation_config": {
            "min_new_tokens": 200,
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "do_sample": True
        },
        "num_samples": args.num_samples,
        "batch_size": args.batch_size
    }
    save_config(paths['config'], config_data)
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(tested_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("/mnt/lustrenew/mllm_safety-shared/datasets/huggingface/allenai/c4/", "default", split="validation", streaming=True)
    column_names =  ['timestamp', 'url']
    
    max_length = 250
    min_length = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    logging.info(f"Using device: {device}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    
    def filter_length(example):
        return len(tokenizer(example['text'], truncation=True, max_length=max_length)["input_ids"]) >= min_length

    def encode(examples):
        trunc_tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
        prompt = tokenizer(
            examples["text"], truncation=True, padding=True, max_length=50, return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        examples["text_completion"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, 50:], skip_special_tokens=True
        )
        return examples

    def move_to_device(batch, device):
        new_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                new_batch[key] = value.to(device)
            elif isinstance(value, list):
                new_batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                new_batch[key] = value
        return new_batch

    # Prepare data
    dataset = dataset.filter(filter_length)
    dataset = dataset.map(encode, batched=True, remove_columns=column_names)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size)

    # Collect prompts and reference data
    prompts = []
    human_text = []
    prompt_text = []
    full_human_text = []
    
    logging.info("Loading and processing dataset...")
    for batch in tqdm(dataloader, desc="Loading data"):
        if len(human_text) >= args.num_samples:
            break
        if isinstance(batch["input_ids"], list):
            batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
        if isinstance(batch["attention_mask"], list):
            batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
        
        prompts.append(batch)
        human_text.extend(batch["text_completion"])
        prompt_text.extend(batch["prompt_text"])
        full_human_text.extend(batch["text"])
    
    human_text = human_text[:args.num_samples]
    prompt_text = prompt_text[:args.num_samples]
    full_human_text = full_human_text[:args.num_samples]   
    logging.info("Data loaded and processed successfully")
    
    logging.info("Loading models...")
    tested_model = AutoModelForCausalLM.from_pretrained(tested_model_name).half()
    watermarked_model = AutoModelForCausalLM.from_pretrained(args.watermark_model).half()
    vanilla_model = AutoModelForCausalLM.from_pretrained(vanilla_model_name).half()
    
    # Create task vector
    task_vector = TaskVector(vanilla_model, watermarked_model)
    logging.info(f"Applying coefficient {args.coefficient}...")
    tested_model.to('cpu')
    coefficient_watermarked_model = task_vector.apply_to(tested_model, scaling_coef=args.coefficient)
    coefficient_watermarked_model.to("cuda")
    vanilla_model.to("cuda")

    # Generate outputs
    logging.info("Generating text samples...")
    vanilla_output_results = []
    watermarked_output_results = []
    
    for batch in tqdm(prompts, desc="Generating"):
        if len(vanilla_output_results) >= args.num_samples:
            break
            
        with torch.no_grad():
            batch = move_to_device(batch, "cuda")
            
            vanilla_output = vanilla_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=config_data["generation_config"]["do_sample"],
                min_new_tokens=config_data["generation_config"]["min_new_tokens"],
                max_new_tokens=config_data["generation_config"]["max_new_tokens"],
                temperature=config_data["generation_config"]["temperature"],
                top_p=config_data["generation_config"]["top_p"],
                top_k=config_data["generation_config"]["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
            
            watermarked_output = coefficient_watermarked_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=config_data["generation_config"]["do_sample"],
                min_new_tokens=config_data["generation_config"]["min_new_tokens"],
                max_new_tokens=config_data["generation_config"]["max_new_tokens"],
                temperature=config_data["generation_config"]["temperature"],
                top_p=config_data["generation_config"]["top_p"],
                top_k=config_data["generation_config"]["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
            n_input_tokens = batch["input_ids"].shape[1]
            vanilla_output_cpu = vanilla_output[:, n_input_tokens:]
            watermarked_output_cpu = watermarked_output[:, n_input_tokens:]
            
            vanilla_output_results.extend(tokenizer.batch_decode(vanilla_output_cpu, skip_special_tokens=True))
            watermarked_output_results.extend(tokenizer.batch_decode(watermarked_output_cpu, skip_special_tokens=True))
    
    vanilla_output_results = vanilla_output_results[:args.num_samples]
    watermarked_output_results = watermarked_output_results[:args.num_samples]

    del coefficient_watermarked_model
    del watermarked_model
    del vanilla_model
    del tested_model
    torch.cuda.empty_cache()

    # Save outputs
    # outputs_data = {
    #     "vanilla_outputs": vanilla_output_results,
    #     "watermarked_outputs": watermarked_output_results,
    #     "prompt_texts": prompt_text,
    #     "human_texts": human_text
    # }
    # save_outputs(paths['outputs'], outputs_data)
    # logging.info("Outputs saved")
    
    # Compute detection scores

    # Compute additional metrics (perplexity and repetition)
    logging.info("Computing additional metrics...")
    metrics_results = compute_metrics_for_outputs(
        vanilla_output_results, 
        watermarked_output_results,
        prompt_text,
        tokenizer,
        args.vanilla_model_name,  # Use tested model for PPL computation
        batch_size=args.batch_size,
        fp16=True
    )

    logging.info("Computing detection scores...")
    detector = WatermarkDetector(
        device=watermark_config.get("kgw_device", 'cpu'),
        tokenizer=tokenizer,
        vocab=tokenizer.get_vocab().values(),
        gamma=watermark_config["gamma"],
        seeding_scheme=watermark_config["seeding_scheme"],
        normalizers=[],
    )

    def compute_p_value(samples, detector):
        score_list = []
        for s in tqdm(samples, desc="Computing p-values"):
            score = detector.detect(s)
            score_list.append(score['p_value'])
        return score_list
    
    vanilla_scores = compute_p_value(vanilla_output_results, detector)
    watermarked_scores = compute_p_value(watermarked_output_results, detector)
    print(f"Vanilla median p-value: {float(np.median(vanilla_scores)):.4f}")
    print(f"Watermarked median p-value: {float(np.median(watermarked_scores)):.4f}")

    
    # Save scores with metadata
    scores_data = {
        "detection_scores": {
            "vanilla_scores": vanilla_scores,
            "watermarked_scores": watermarked_scores,
            "mean_vanilla_score": float(np.mean(vanilla_scores)),
            "mean_watermarked_score": float(np.mean(watermarked_scores)),
            "median_watermarked_score": float(np.median(watermarked_scores)),
            "median_vanilla_score": float(np.median(vanilla_scores))
        },
        "perplexity_metrics": {
            "vanilla": {
                "mean_perplexity": metrics_results["vanilla"]["mean_perplexity"],
                "median_perplexity": metrics_results["vanilla"]["median_perplexity"],
                "perplexities": metrics_results["vanilla"]["perplexities"]
            },
            "watermarked": {
                "mean_perplexity": metrics_results["watermarked"]["mean_perplexity"],
                "median_perplexity": metrics_results["watermarked"]["median_perplexity"],
                "perplexities": metrics_results["watermarked"]["perplexities"]
            }
        },
        "repetition_metrics": {
            "vanilla": {
                "median_seq_rep_3": metrics_results["vanilla"]["median_seq_rep_3"],
                "mean_seq_rep_3": metrics_results["vanilla"]["mean_seq_rep_3"],
                "total_rep_3": metrics_results["vanilla"]["total_rep_3"],
                "list_seq_rep_3": metrics_results["vanilla"]["list_seq_rep_3"]
            },
            "watermarked": {
                "median_seq_rep_3": metrics_results["watermarked"]["median_seq_rep_3"],
                "mean_seq_rep_3": metrics_results["watermarked"]["mean_seq_rep_3"],
                "total_rep_3": metrics_results["watermarked"]["total_rep_3"],
                "list_seq_rep_3": metrics_results["watermarked"]["list_seq_rep_3"]
            }
        },
        "metadata": {
            "detection_method": "kgw",
            "num_samples": len(vanilla_scores),
            "ppl_model": args.tested_model_name
        }
    }
    save_scores(paths['scores'], scores_data)
    
    # Cleanup
    logging.info(f"Completed processing {watermark_checkpoint_name} coefficient {args.coefficient}")
    logging.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()