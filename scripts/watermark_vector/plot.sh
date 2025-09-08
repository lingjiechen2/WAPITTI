# Basic usage
python experiments/plot.py \
    --tested_model "Meta_Llama_3.1_8B" \
    --watermark_name "Meta_Llama_3.1_8B_logit_watermark_distill_kgw_k0_gamma0.25_delta2"

# # With custom options
# python experiments/plot.py \
#     --base_dir "results/watermark_vectors" \
#     --tested_model "Meta_Llama_3.1_8B" \
#     --watermark_name "Meta_Llama_3.1_8B_logit_watermark_distill_kgw_k1_gamma0.25_delta1" \
#     --output_dir "my_plots" \
#     --figsize 18 15