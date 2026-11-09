#!/bin/bash
# Quick test (~10-15 min): GPU visible? do both readers run on 20 charts? Submit with:
#   bash slurm/sbatch.sh slurm/smoke.sbatch
#SBATCH --job-name=ca-smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
M=data/synthetic_v2/val/manifest.jsonl
gpu_py src/run_reader.py --reader qwen   --manifest $M --limit 20 --n 3 --temperature 0.7 --out outputs/readings/smoke_qwen.jsonl
gpu_py src/run_reader.py --reader deplot --manifest $M --limit 20 --n 3 --temperature 0.7 --out outputs/readings/smoke_deplot.jsonl
gpu_py src/run_reader.py --reader qwen   --manifest $M --limit 20 --task judge --out outputs/readings/smoke_b0.jsonl
cpu_py src/evaluate.py --manifest $M --readings outputs/readings/smoke_qwen.jsonl
cpu_py src/evaluate.py --manifest $M --readings outputs/readings/smoke_deplot.jsonl
cpu_py src/evaluate.py --manifest $M --readings outputs/readings/smoke_b0.jsonl
