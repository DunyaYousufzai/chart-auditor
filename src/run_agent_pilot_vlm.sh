#!/usr/bin/env bash
#SBATCH --job-name=chart_agent_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/agent_pilot_%j.out
#SBATCH --error=outputs/logs/agent_pilot_%j.err

set -euo pipefail
mkdir -p outputs/logs outputs/results

# Activa tu entorno Python / conda / venv en UBELIX
# source /path/to/your/venv/bin/activate

# Mantén cachés de HuggingFace / Torch en scratch para evitar llenar $HOME
export HF_HOME="${SCRATCH:-$HOME/.cache}/hf_cache"
export TORCH_HOME="${SCRATCH:-$HOME/.cache}/torch_cache"

python src/run_agent_pilot.py \
  --manifest data/synthetic_v2/test_unseen/manifest.jsonl \
  --reader qwen \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --limit 30 \
  --n-initial 3 \
  --n-retry 2 \
  --min-side 800 \
  --out "outputs/results/agent_qwen_unseen_${SLURM_JOB_ID}.jsonl"