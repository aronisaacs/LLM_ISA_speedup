#!/usr/bin/env bash
# Manual launch only. Independent K and V calibration, then both greedy studies.
# Screen: 16 random chunks. Refine: best two settings on 32 different chunks.
# Close/reordered comparisons: 32 extra chunks. Final C-Eval: full valid split.
# Preview without running: python compression_topics/spatial/scripts/pair_calibrated_study.py
# Run everything: sbatch compression_topics/spatial/scripts/pair_calibrated_dgx.sh
# Calibration only: sbatch .../pair_calibrated_dgx.sh --through select
# Resume after calibration: sbatch .../pair_calibrated_dgx.sh --from greedy
#SBATCH --job-name=spatial-calibrated
#SBATCH --partition=prod
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=192
#SBATCH --mem=600000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_calibrated_%j.out

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
python -u compression_topics/spatial/scripts/pair_calibrated_study.py --execute "$@"
