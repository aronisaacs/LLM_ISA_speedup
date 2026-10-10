#!/usr/bin/env bash
#SBATCH --job-name=spatial-online-pairs
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:20:00
#SBATCH --output=logs/spatial_online_pairs_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
echo "Job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_online_pairs
python -u compression_topics/spatial/scripts/online_pairs_quick.py "$@"
