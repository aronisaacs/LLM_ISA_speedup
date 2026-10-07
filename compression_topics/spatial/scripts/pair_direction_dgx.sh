#!/usr/bin/env bash
# Run manually on the DGX. Default: fresh 50-chunk cosine profile only.
# Submit through Slurm so it allocates a free GPU; do not set GPU IDs yourself.
# sbatch compression_topics/spatial/scripts/pair_direction_dgx.sh
# Optional accuracy studies (each is a separate long job):
# sbatch compression_topics/spatial/scripts/pair_direction_dgx.sh merge
# sbatch compression_topics/spatial/scripts/pair_direction_dgx.sh residual
#SBATCH --job-name=spatial-direction
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_direction_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export PYTHONUNBUFFERED=1

case "${1:-profile}" in
  profile)
    python -u compression_topics/spatial/scripts/pair_direction_profile.py \
      --samples 50 --seq-len 2048 --seed 0
    ;;
  merge|residual)
    python -u compression_topics/spatial/scripts/pair_rank_study.py \
      --direction --variant "$1"
    ;;
  *)
    echo 'Usage: sbatch pair_direction_dgx.sh [profile|merge|residual]' >&2
    exit 2
    ;;
esac
