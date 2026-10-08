#!/usr/bin/env bash
# Submit from the repository root after mkdir -p logs.
# Stage A for group_rd: one prefill of the 16 calibration chunks (keys, values,
# query second moments; about 4 GB under figures/group_rd_offline/keys, which
# git ignores), then the offline menu analysis on CPU. No perplexity runs.
# Results: compression_topics/spatial/figures/group_rd_offline/stage_a.{json,md}.
# sbatch compression_topics/spatial/scripts/group_rd_offline_dgx.sh
# Analysis only, reusing captured keys: sbatch .../group_rd_offline_dgx.sh --stage analyze
#SBATCH --job-name=spatial-rd-stage-a
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_rd_stage_a_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export PYTHONUNBUFFERED=1
echo "Job: ${SLURM_JOB_ID:-manual}; GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_group_rd
python -u compression_topics/spatial/scripts/group_rd_offline.py --execute --stage all --device cuda "$@"
