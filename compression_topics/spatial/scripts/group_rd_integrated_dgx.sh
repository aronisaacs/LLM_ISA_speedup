#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-integrated
#SBATCH --partition=prod
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=192
#SBATCH --mem=600000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_rd_integrated_%j.out
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
python -m unittest unit_tests.test_group_rd unit_tests.test_group_rd_integrated
python -u compression_topics/spatial/scripts/group_rd_integrated_study.py --execute "$@"
