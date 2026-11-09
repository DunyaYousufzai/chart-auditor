#!/bin/bash
# Submit every reading job of the pilot. Run on the login node from the repo root:
#   bash slurm/pilot.sh
# Then, when they finish:  bash slurm/pilot_eval.sh
set -euo pipefail
Q=Qwen/Qwen3-VL-4B-Instruct
R=outputs/readings
for SET in synthetic_v2/val synthetic_v2/test_seen synthetic_v2/test_unseen chartqa_pairs/val chartqa_pairs/test; do
  M=data/$SET/manifest.jsonl
  TAG=$(echo $SET | tr '/' '_')
  # our reader: Qwen3-VL-4B, 9 readings (evaluate.py can use the first 1/3/5/9)
  bash slurm/sbatch.sh -J ca-qwen-$TAG slurm/read.sbatch qwen $M $R/qwen4b_${TAG}_n9.jsonl --model $Q --n 9 --temperature 0.7
  # baseline B1: DePlot, greedy + 5 samples
  bash slurm/sbatch.sh -J ca-deplot-$TAG slurm/read.sbatch deplot $M $R/deplot_${TAG}_n5.jsonl --n 5 --temperature 0.7
  # baseline B0: Qwen sees chart + paper table and answers MATCH/MISMATCH
  bash slurm/sbatch.sh -J ca-b0-$TAG slurm/read.sbatch qwen $M $R/b0_qwen4b_${TAG}.jsonl --model $Q --task judge
done
squeue -u "$USER"
