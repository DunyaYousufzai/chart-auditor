#!/bin/bash
# Submit the pilot: one job per reader, each reads all 5 sets (model loaded once).
#   bash slurm/pilot.sh          then, when done:   bash slurm/pilot_eval.sh
set -euo pipefail
Q=Qwen/Qwen3-VL-4B-Instruct
R=outputs/readings
SETS="synthetic_v2/val synthetic_v2/test_seen synthetic_v2/test_unseen chartqa_pairs/val chartqa_pairs/test"
M=(); OQ=(); OD=(); OB=()
for SET in $SETS; do
  TAG=$(echo $SET | tr '/' '_')
  M+=(data/$SET/manifest.jsonl)
  OQ+=($R/qwen4b_${TAG}_n9.jsonl)     # our reader: Qwen3-VL-4B, 9 readings (evaluate can use 1/3/5/9)
  OD+=($R/deplot_${TAG}_n5.jsonl)     # baseline B1: DePlot, 5 readings
  OB+=($R/b0_qwen4b_${TAG}.jsonl)     # baseline B0: Qwen sees chart + table, answers MATCH/MISMATCH
done
bash slurm/sbatch.sh -J ca-qwen   slurm/read.sbatch qwen   --manifest "${M[@]}" --out "${OQ[@]}" --model $Q --n 9 --temperature 0.7
bash slurm/sbatch.sh -J ca-deplot slurm/read.sbatch deplot --manifest "${M[@]}" --out "${OD[@]}" --n 5 --temperature 0.7 --batch 16
bash slurm/sbatch.sh -J ca-b0     slurm/read.sbatch qwen   --manifest "${M[@]}" --out "${OB[@]}" --model $Q --task judge
squeue -u "$USER"
