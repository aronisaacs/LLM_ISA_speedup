#!/usr/bin/env bash
#SBATCH --job-name=spatial-clean-ladder
#SBATCH --partition=prod
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=192
#SBATCH --mem=600000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_clean_ladder_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
echo "Job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_presentation_ladder
python -u compression_topics/spatial/scripts/presentation_ladder_study.py --execute "$@"
