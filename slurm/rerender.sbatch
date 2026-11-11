#!/bin/bash
# Re-render DaTikZ test and val charts at high resolution (CPU only). Usage:
#   sbatch --nodes=1 --partition=epyc2 --account=gratis --qos=job_cpu slurm/rerender.sbatch [size]
#SBATCH --job-name=ca-rerender
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
# Python first, then TeX: loading the Python module after texlive would unload texlive
module load "$PY_MODULE" texlive/20230313-GCC-12.3.0 poppler/23.09.0-GCC-12.3.0
SIZE=${1:-1260}
for S in test val; do
  "$CPU_VENV/bin/python" src/rerender.py --manifest data/datikz_pgf/$S/manifest.jsonl --split-out ${S}_hr --size "$SIZE" --jobs 16
done
