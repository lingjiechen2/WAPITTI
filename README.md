# WAPITTI
Code repository for WAPITTI 


## Watermark distillation training scripts



## Watermark vector evaluation scripts

[`scripts/watermark_vector/evaluate.sh`](scripts/watermark_vector/evaluate.sh) runs evaluation on watermark-distilled model's watermark. For each watermark-distilled model, it will extract the watermark vector from all checkpoints, and apply watermark insertion into target tested model. For each insertion coefficient, the model will generate samples and carry out evaluation including ppl, p-value and sep-rep on all inference results.

