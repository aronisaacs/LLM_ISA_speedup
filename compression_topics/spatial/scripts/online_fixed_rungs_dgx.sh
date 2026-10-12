#!/usr/bin/env bash
#SBATCH --job-name=spatial-fixed-rungs
#SBATCH --partition=prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=24:00:00
#SBATCH --output=logs/spatial_fixed_rungs_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "Slurm job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_online_fixed_rungs unit_tests.test_per_head_decisions unit_tests.test_query_absolute_error
python -u compression_topics/spatial/scripts/online_fixed_rungs.py --execute "$@"
