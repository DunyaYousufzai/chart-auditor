#!/bin/bash
# Re-render DaTikZ test and val charts at high resolution (CPU only). Usage:
#   sbatch --nodes=1 --partition=epyc2 --account=gratis --qos=job_gratis slurm/rerender.sbatch [size]
# Optional env vars: SPLITS (default "test val"), FIGS (TikZ sources; train uses data/datikz_pgf/figs_all).
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
FIGS=${FIGS:-data/datikz_pgf/figs}
for S in ${SPLITS:-test val}; do
  "$CPU_VENV/bin/python" src/rerender.py --manifest data/datikz_pgf/$S/manifest.jsonl --figs "$FIGS" --split-out ${S}_hr --size "$SIZE" --jobs 16
done
