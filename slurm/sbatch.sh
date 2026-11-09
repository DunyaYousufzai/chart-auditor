#!/bin/bash
# sbatch with our account/partition/QoS from site.env. Usage: bash slurm/sbatch.sh [sbatch args] script.sbatch [script args]
set -a; source "$(dirname "$0")/site.env"; set +a
exec sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" --qos="$GPU_QOS" --gres="$GPU_GRES" "$@"
