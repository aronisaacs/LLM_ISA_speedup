#!/usr/bin/env bash
# Submit from the repository root after mkdir -p logs.
# Stage B for group_rd: C-Eval (5-shot) and WikiText with per-block formats on
# keys and values, 30/40/50% saving in every layer, plain vs attention-aware
# key residuals. One worker per granted GPU. Rerunning skips finished runs.
# Results: compression_topics/spatial/figures/group_rd_tasks/summary.md and results.json.
# sbatch compression_topics/spatial/scripts/group_rd_tasks_dgx.sh
# Another menu:   sbatch .../group_rd_tasks_dgx.sh --menu default
#SBATCH --job-name=spatial-rd-tasks
#SBATCH --partition=prod
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=192
#SBATCH --mem=600000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_rd_tasks_%j.out

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
python -m unittest unit_tests.test_group_rd
python -u compression_topics/spatial/scripts/group_rd_tasks_study.py --execute "$@"
