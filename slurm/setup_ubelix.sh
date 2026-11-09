#!/bin/bash
# Run ONCE on the UBELIX login node, from the repo root (~/chart-auditor).
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
echo "code:  $(pwd)"
echo "store: $PROJ_STORE"

echo "[1/5] folders"
# 1. Folders. data/* and outputs/* are in .gitignore, so git does not see these links.
mkdir -p "$PROJ_STORE"/{images,pylib,hf_cache,vllm_cache,data/synthetic_v2,data/chartqa_raw,data/chartqa_pairs,outputs/readings,outputs/results}
for d in data/synthetic_v2 data/chartqa_raw data/chartqa_pairs outputs/readings outputs/results; do
  [ -e "$d" ] || ln -s "$PROJ_STORE/$d" "$d"
done

echo "[2/5] vLLM container"
# 2. vLLM container: reuse the one built for spec-compression (hard link = no extra space,
#    and it survives if that project deletes its copy).
SRC_SIF=/storage/scratch/users/$USER/spec-compression/images/vllm.sif
if [ ! -f "$VLLM_SIF" ]; then
  ln "$SRC_SIF" "$VLLM_SIF" 2>/dev/null || cp "$SRC_SIF" "$VLLM_SIF"
fi

# peft is not in the container: install it next to it, without deps (keeps vLLM's torch)
apptainer exec --cleanenv -B "$PROJ_STORE" "$VLLM_SIF" python3 -m pip install -q --no-deps --target "$PROJ_STORE/pylib" peft

echo "[3/5] CPU python env"
# 3. Small CPU env for data generation and evaluation
module load "$PY_MODULE"
[ -d "$CPU_VENV" ] || python3 -m venv "$CPU_VENV"
"$CPU_VENV/bin/pip" install -q --upgrade pip
"$CPU_VENV/bin/pip" install -q -r requirements.txt

echo "[4/5] data: synthetic + ChartQA"
# 4. Data. The generator is deterministic: we rebuild it instead of copying it.
[ -f data/synthetic_v2/test_unseen/manifest.jsonl ] || cpu_py src/generate_dataset.py --out data/synthetic_v2
if [ ! -d "data/chartqa_raw/ChartQA Dataset" ]; then
  curl -L --progress-bar --retry 5 -C - -o data/chartqa_raw/chartqa.zip \
    "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
  (cd data/chartqa_raw && unzip -q chartqa.zip && rm chartqa.zip)
fi
cpu_py src/chartqa_pairs.py --split test --limit 1000
cpu_py src/chartqa_pairs.py --split val --limit 500

echo "[5/5] model weights"
# 5. Model weights, so jobs do not wait for them
apptainer exec --cleanenv --env HF_HOME="$HF_HOME" -B "$PROJ_STORE" "$VLLM_SIF" python3 -c "
from huggingface_hub import snapshot_download
for m in ['Qwen/Qwen3-VL-4B-Instruct', 'google/deplot']:
    print('downloading', m); snapshot_download(m)"
echo "Setup done. Next: bash slurm/sbatch.sh slurm/smoke.sbatch"
