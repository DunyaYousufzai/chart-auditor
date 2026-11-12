# Sourced by every script/job, from the repo root.
set -a; source slurm/site.env; set +a
export HF_HOME="$PROJ_STORE/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

# gpu_py ...  : run python inside the vLLM container with the GPU (for readers / training).
#              Extra packages missing from the container (peft) live in $PROJ_STORE/pylib.
gpu_py() {
  apptainer exec --nv --cleanenv \
    --env HF_HOME="$HF_HOME",PYTHONPATH="$PROJ_STORE/pylib",PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false,VLLM_CACHE_ROOT="$PROJ_STORE/vllm_cache" \
    -B "$PROJ_STORE" -B "$PWD" --pwd "$PWD" "$VLLM_SIF" python3 "$@"
}

# cpu_py ...  : plain python with matplotlib/pandas/scipy (data generation, evaluation)
cpu_py() {
  module load "$PY_MODULE" >/dev/null 2>&1 || true
  "$CPU_VENV/bin/python" "$@"
}
