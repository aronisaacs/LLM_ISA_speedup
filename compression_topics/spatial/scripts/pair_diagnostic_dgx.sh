#!/usr/bin/env bash
#SBATCH --job-name=pair-diagnostic
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --output=logs/pair_diagnostic_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
python -m unittest unit_tests.test_storage_contract unit_tests.test_spatial_recorded_results unit_tests.test_pair_diagnostic
python -u compression_topics/spatial/scripts/pair_diagnostic.py --execute "$@"
