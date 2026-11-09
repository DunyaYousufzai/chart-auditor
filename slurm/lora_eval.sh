#!/bin/bash
# After a LoRA run: read the 5 pilot sets with the merged model (one GPU job, like pilot.sh).
#   bash slurm/lora_eval.sh <run_name>        then evaluate like the pilot (pilot_eval.sh pattern)
# To chain it after training:  bash slurm/lora_eval.sh <run_name> --dependency=afterok:<train_job_id>
set -euo pipefail
RUN=$1; shift
R=outputs/readings
SETS="synthetic_v2/val synthetic_v2/test_seen synthetic_v2/test_unseen chartqa_pairs/val chartqa_pairs/test"
M=(); O=()
for SET in $SETS; do
  TAG=$(echo $SET | tr '/' '_')
  M+=(data/$SET/manifest.jsonl); O+=($R/lora_${RUN}_${TAG}_n9.jsonl)
done
bash slurm/sbatch.sh -J ca-lora-read "$@" slurm/read.sbatch qwen --model outputs/lora/$RUN/merged \
  --manifest "${M[@]}" --out "${O[@]}" --n 9 --temperature 0.7
