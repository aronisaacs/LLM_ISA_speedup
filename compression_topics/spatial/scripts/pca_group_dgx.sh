#!/usr/bin/env bash
# Both PCA options + independent residual baseline, four-token groups only.
# One GPU collects samples; screen/refine/tasks use all three granted GPUs.
# Optional: --options rank1 OR --options shared (use different --out folders).
# Resume evaluations: --from screen; calibration only: --through calibrate.
#SBATCH --job-name=spatial-pca-quads
#SBATCH --partition=prod
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=96
#SBATCH --mem=300000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/spatial_pca_quads_%j.out
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
python -m unittest unit_tests.test_pca_group
python -u compression_topics/spatial/scripts/pca_group_study.py --execute "$@"
