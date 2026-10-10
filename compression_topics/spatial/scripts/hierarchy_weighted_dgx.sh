#!/usr/bin/env bash
#SBATCH --job-name=spatial-weighted-g8
#SBATCH --partition=prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --time=24:00:00
#SBATCH --output=logs/spatial_weighted_g8_%j.out
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
python -m unittest unit_tests.test_spatial_hierarchy unit_tests.test_hierarchy_weighted unit_tests.test_calibration_study
python -u compression_topics/spatial/scripts/hierarchy_weighted.py --execute "$@"
