#!/usr/bin/env bash
# Submit from the repository root after mkdir -p logs.
# Per layer/K/V slot: choose pairs or quads using matched calibration chunks.
# Final evaluation: pairs-only versus mixed slot assignments.
# Budgets include residual masks, norms, merge flags, and slot headers.
# Calibration only: sbatch .../group_calibrated_dgx.sh --through select
# Resume tasks: sbatch .../group_calibrated_dgx.sh --from greedy
#SBATCH --job-name=spatial-groups
#SBATCH --partition=prod
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=96
#SBATCH --mem=300000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_groups_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
echo "Job: ${SLURM_JOB_ID:-manual}; GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_group_calibrated unit_tests.test_pair_calibrated
python -u compression_topics/spatial/scripts/group_calibrated_study.py --execute "$@"
