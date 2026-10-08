#!/usr/bin/env bash
# Submit from the repository root after mkdir -p logs.
# Shared signed pair residual; independent quad residuals.
# Per layer/K/V slot: choose pairs or quads using matched calibration chunks.
# Final evaluation: pairs-only versus mixed slot assignments.
# Final C-Eval scores go to the repository results.json.
# New output directory: compression_topics/spatial/figures/pair_quad_shared_sampled.
# Budgets include residual masks, norms, merge flags, and slot headers.
# Calibration only: sbatch .../group_calibrated_dgx.sh --through select
# Resume tasks: sbatch .../group_calibrated_dgx.sh --from greedy
#SBATCH --job-name=spatial-pairs-quads
#SBATCH --partition=prod
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=192
#SBATCH --mem=600000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_pairs_quads_%j.out

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
# Remove obsolete untracked DGX artifacts that a Git pull cannot delete.
# These directories contain only the abandoned duplicated-pair/PCA studies.
rm -rf -- compression_topics/spatial/figures/pair_quad_metadata_sampled \
          compression_topics/spatial/figures/pca_quad_heads \
          compression_topics/spatial/figures/pca_layer16_profile
python -m unittest unit_tests.test_group_calibrated unit_tests.test_pair_calibrated
python -u compression_topics/spatial/scripts/group_calibrated_study.py --execute "$@"
