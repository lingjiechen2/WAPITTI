import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Plot watermark vector results")
    parser.add_argument("--base_dir", type=str, 
                       default="results/watermark_vectors",
                       help="Base directory containing results")
    parser.add_argument("--tested_model", type=str, required=True,
                       help="Tested model name (e.g., Meta_Llama_3.1_8B)")
    parser.add_argument("--watermark_name", type=str, required=True,
                       help="Watermark name (e.g., Meta_Llama_3.1_8B_logit_watermark_distill_kgw_k1_gamma0.25_delta1)")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--figsize", nargs=2, type=int, default=[15, 12],
                       help="Figure size (width, height)")
    return parser.parse_args()

def extract_coefficient_from_path(coeff_dir):
    """Extract coefficient value from directory name like 'coefficient_1.5'"""
    match = re.search(r'coefficient_(-?\d+\.?\d*)', coeff_dir)
    if match:
        return float(match.group(1))
    return None

def extract_checkpoint_number(checkpoint_dir):
    """Extract checkpoint number from directory name like 'checkpoint_3000'"""
    match = re.search(r'checkpoint_(\d+)', checkpoint_dir)
    if match:
        return int(match.group(1))
    return None

def load_all_results(base_dir, tested_model, watermark_name):
    """Load all results for given model and watermark combination"""
    model_dir = os.path.join(base_dir, tested_model, watermark_name)
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    
    results = defaultdict(dict)  # {checkpoint: {coefficient: data}}
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('checkpoint_')]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_num = extract_checkpoint_number(checkpoint_dir)
        if checkpoint_num is None:
            continue
            
        checkpoint_path = os.path.join(model_dir, checkpoint_dir)
        
        # Find all coefficient directories
        coeff_dirs = [d for d in os.listdir(checkpoint_path)
                     if os.path.isdir(os.path.join(checkpoint_path, d)) and d.startswith('coefficient_')]
        
        for coeff_dir in coeff_dirs:
            coeff_value = extract_coefficient_from_path(coeff_dir)
            if coeff_value is None:
                continue
                
            scores_path = os.path.join(checkpoint_path, coeff_dir, 'scores.json')
            
            if os.path.exists(scores_path):
                try:
                    with open(scores_path, 'r') as f:
                        scores_data = json.load(f)
                    results[checkpoint_num][coeff_value] = scores_data
                except Exception as e:
                    print(f"Error loading {scores_path}: {e}")
    
    return results

def prepare_plot_data(results):
    """Prepare data for plotting"""
    plot_data = defaultdict(lambda: defaultdict(list))  # {checkpoint: {metric: [(coeff, value)]}}
    
    for checkpoint, coeff_data in results.items():
        for coeff, data in coeff_data.items():
            # Extract metrics
            if 'detection_scores' in data:
                det_scores = data['detection_scores']
                plot_data[checkpoint]['vanilla_p_value'].append((coeff, det_scores.get('median_vanilla_score', 0)))
                plot_data[checkpoint]['watermarked_p_value'].append((coeff, det_scores.get('median_watermarked_score', 0)))
            
            if 'perplexity_metrics' in data:
                ppl_metrics = data['perplexity_metrics']
                if 'vanilla' in ppl_metrics:
                    plot_data[checkpoint]['vanilla_perplexity'].append((coeff, ppl_metrics['vanilla'].get('median_perplexity', 0)))
                if 'watermarked' in ppl_metrics:
                    plot_data[checkpoint]['watermarked_perplexity'].append((coeff, ppl_metrics['watermarked'].get('median_perplexity', 0)))
            
            if 'repetition_metrics' in data:
                rep_metrics = data['repetition_metrics']
                if 'vanilla' in rep_metrics:
                    plot_data[checkpoint]['vanilla_rep3'].append((coeff, rep_metrics['vanilla'].get('median_seq_rep_3', 0)))
                if 'watermarked' in rep_metrics:
                    plot_data[checkpoint]['watermarked_rep3'].append((coeff, rep_metrics['watermarked'].get('median_seq_rep_3', 0)))
    
    # Sort by coefficient for each metric
    for checkpoint in plot_data:
        for metric in plot_data[checkpoint]:
            plot_data[checkpoint][metric].sort(key=lambda x: x[0])
    
    return plot_data

def create_comprehensive_plot(plot_data, tested_model, watermark_name, figsize):
    """Create comprehensive plot with multiple subplots"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Watermark Vector Analysis\nModel: {tested_model}\nWatermark: {watermark_name}', 
                 fontsize=14, fontweight='bold')
    
    # Define colors for different checkpoints
    checkpoints = sorted(plot_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoints)))
    
   # Subplot configurations - WATERMARKED metrics on first row, VANILLA on second row
    subplot_configs = [
        ('watermarked_p_value', 'Watermarked P-Value', axes[0, 0]),
        ('watermarked_perplexity', 'Watermarked Perplexity', axes[0, 1]),
        ('watermarked_rep3', 'Watermarked Seq-Rep-3', axes[0, 2]),
        ('vanilla_p_value', 'Vanilla P-Value', axes[1, 0]),
        ('vanilla_perplexity', 'Vanilla Perplexity', axes[1, 1]),
        ('vanilla_rep3', 'Vanilla Seq-Rep-3', axes[1, 2])
    ]
    
    for metric, title, ax in subplot_configs:
        for i, checkpoint in enumerate(checkpoints):
            if metric in plot_data[checkpoint] and plot_data[checkpoint][metric]:
                coeffs, values = zip(*plot_data[checkpoint][metric])
                ax.plot(coeffs, values, 'o-', color=colors[i], 
                       label=f'Checkpoint {checkpoint}', linewidth=2, markersize=4)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_ylabel(title.split()[-1], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Special formatting for p-value plots (log scale if needed)
        if 'p_value' in metric:
            ax.set_yscale('log')
            ax.set_ylabel('P-Value (log scale)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def save_summary_stats(plot_data, output_path):
    """Save summary statistics to a text file"""
    with open(output_path, 'w') as f:
        f.write("Watermark Vector Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for checkpoint in sorted(plot_data.keys()):
            f.write(f"Checkpoint {checkpoint}:\n")
            f.write("-" * 20 + "\n")
            
            for metric in plot_data[checkpoint]:
                if plot_data[checkpoint][metric]:
                    coeffs, values = zip(*plot_data[checkpoint][metric])
                    f.write(f"  {metric}:\n")
                    f.write(f"    Coefficient range: {min(coeffs):.1f} to {max(coeffs):.1f}\n")
                    f.write(f"    Value range: {min(values):.4f} to {max(values):.4f}\n")
                    f.write(f"    Mean value: {np.mean(values):.4f}\n")
            f.write("\n")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results for {args.tested_model} / {args.watermark_name}...")
    
    try:
        # Load all results
        results = load_all_results(args.base_dir, args.tested_model, args.watermark_name)
        
        if not results:
            print("No results found!")
            return
        
        print(f"Found data for {len(results)} checkpoints")
        
        # Prepare plot data
        plot_data = prepare_plot_data(results)
        
        # Create plot
        fig = create_comprehensive_plot(plot_data, args.tested_model, args.watermark_name, args.figsize)
        
        # Save plot
        plot_filename = f"{args.tested_model}_{args.watermark_name}_analysis.png"
        plot_path = os.path.join(args.output_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        
        # Save summary statistics
        summary_filename = f"{args.tested_model}_{args.watermark_name}_summary.txt"
        summary_path = os.path.join(args.output_dir, summary_filename)
        save_summary_stats(plot_data, summary_path)
        print(f"Summary saved to: {summary_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()