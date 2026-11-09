#!/bin/bash
# Evaluate the pilot readings (CPU only: run on the login node or locally after rsync).
set -euo pipefail
source slurm/env.sh
R=outputs/readings; O=outputs/results
for SRC in synthetic_v2 chartqa_pairs; do
  for TEST in $( [ $SRC = synthetic_v2 ] && echo "test_seen test_unseen" || echo "test" ); do
    for N in 1 3 5 9; do
      cpu_py src/evaluate.py --manifest data/$SRC/$TEST/manifest.jsonl --readings $R/qwen4b_${SRC}_${TEST}_n9.jsonl \
        --val-manifest data/$SRC/val/manifest.jsonl --val-readings $R/qwen4b_${SRC}_val_n9.jsonl \
        --n-use $N --out $O/qwen4b_${SRC}_${TEST}_n$N.json
    done
    cpu_py src/evaluate.py --manifest data/$SRC/$TEST/manifest.jsonl --readings $R/qwen4b_${SRC}_${TEST}_n9.jsonl \
      --val-manifest data/$SRC/val/manifest.jsonl --val-readings $R/qwen4b_${SRC}_val_n9.jsonl \
      --early-stop --out $O/qwen4b_${SRC}_${TEST}_early.json
    cpu_py src/evaluate.py --manifest data/$SRC/$TEST/manifest.jsonl --readings $R/deplot_${SRC}_${TEST}_n5.jsonl \
      --val-manifest data/$SRC/val/manifest.jsonl --val-readings $R/deplot_${SRC}_val_n5.jsonl \
      --out $O/deplot_${SRC}_${TEST}_n5.json
    cpu_py src/evaluate.py --manifest data/$SRC/$TEST/manifest.jsonl --readings $R/b0_qwen4b_${SRC}_${TEST}.jsonl \
      --out $O/b0_qwen4b_${SRC}_${TEST}.json
  done
done
cpu_py src/summarize.py $O | tee $O/summary.md
