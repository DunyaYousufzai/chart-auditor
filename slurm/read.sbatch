#!/bin/bash
# One reader on one manifest. Usage:
#   bash slurm/sbatch.sh slurm/read.sbatch <reader> <manifest> <out.jsonl> [run_reader.py args]
# e.g.
#   bash slurm/sbatch.sh slurm/read.sbatch qwen data/synthetic_v2/val/manifest.jsonl outputs/readings/qwen4b_synthetic_v2_val_n9.jsonl --n 9 --temperature 0.7
#SBATCH --job-name=ca-read
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
READER=$1; MANIFEST=$2; OUT=$3; shift 3
nvidia-smi --query-gpu=name,memory.total --format=csv
gpu_py src/run_reader.py --reader "$READER" --manifest "$MANIFEST" --out "$OUT" "$@"
