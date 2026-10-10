#!/usr/bin/env bash
#SBATCH --job-name=spatial-tree-pilot
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=00:20:00
#SBATCH --output=logs/spatial_tree_pilot_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
python -m unittest unit_tests.test_spatial_hierarchy unit_tests.test_hierarchy_pilot
python -u compression_topics/spatial/scripts/hierarchy_pilot.py "$@"
