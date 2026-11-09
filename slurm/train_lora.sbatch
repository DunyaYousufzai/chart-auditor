#!/bin/bash
# LoRA fine-tune of Qwen3-VL-4B, then merge the adapter. Usage:
#   bash slurm/sbatch.sh --gres=gpu:h100:1 slurm/train_lora.sbatch <run_name> [train_lora.py args]
# e.g. a 20-step smoke test:
#   bash slurm/sbatch.sh --gres=gpu:h100:1 --time=00:40:00 slurm/train_lora.sbatch smoke \
#        --limit-train 200 --limit-val 16 --max-steps 20 --eval-every 10
#SBATCH --job-name=ca-lora
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
RUN=$1; shift
OUT=outputs/lora/$RUN
mkdir -p "$PROJ_STORE/outputs/lora"; [ -e outputs/lora ] || ln -s "$PROJ_STORE/outputs/lora" outputs/lora
nvidia-smi --query-gpu=name,memory.total --format=csv
gpu_py src/train_lora.py --out "$OUT" \
  --train data/synthetic_v2/train/manifest.jsonl data/chartqa_pairs/train/manifest.jsonl \
  --val   data/synthetic_v2/val/manifest.jsonl   data/chartqa_pairs/val/manifest.jsonl \
  "$@"
gpu_py src/merge_lora.py --adapter "$OUT/adapter" --out "$OUT/merged"
