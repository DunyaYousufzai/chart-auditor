#!/bin/bash
# One reader on one or more manifests. Usage:
#   bash slurm/sbatch.sh slurm/read.sbatch <reader> --manifest m1 [m2 ...] --out o1 [o2 ...] [run_reader.py args]
# e.g.
#   bash slurm/sbatch.sh slurm/read.sbatch qwen --manifest data/synthetic_v2/val/manifest.jsonl --out outputs/readings/qwen4b_synthetic_v2_val_n9.jsonl --n 9 --temperature 0.7
#SBATCH --job-name=ca-read
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
READER=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv
gpu_py src/run_reader.py --reader "$READER" "$@"
