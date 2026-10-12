#!/usr/bin/env bash
# sbatch compression_topics/vector/scripts/net_budget_dgx.sh pilot   # 50% selections, a few items per task
# sbatch compression_topics/vector/scripts/net_budget_dgx.sh         # all 45 runs, then the table
# sbatch compression_topics/vector/scripts/net_budget_dgx.sh query   # query-aware key sweeps, greedy, tasks, table
#SBATCH --job-name=vector-net-budget
#SBATCH --partition=prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=24:00:00
#SBATCH --output=logs/vector_net_budget_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "Slurm job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}"
git log -1 --oneline
python -m unittest unit_tests.test_net_budget unit_tests.test_vector_study unit_tests.test_vector_query
if [[ "${1:-full}" == "pilot" ]]; then
  python -u compression_topics/vector/scripts/net_budget_study.py --only pilot
elif [[ "${1:-full}" == "query" ]]; then
  python -u compression_topics/vector/scripts/net_budget_study.py --arms query --from sweep
else
  python -u compression_topics/vector/scripts/net_budget_study.py --from tasks
fi
