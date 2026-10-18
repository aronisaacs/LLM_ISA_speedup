#!/usr/bin/env bash
# sbatch compression_topics/vector/scripts/model_budget_dgx.sh qwen pilot       # quick check, writes nothing
# sbatch compression_topics/vector/scripts/model_budget_dgx.sh qwen             # sweep, greedy, tasks, table
# sbatch compression_topics/vector/scripts/model_budget_dgx.sh codellama pilot
# sbatch compression_topics/vector/scripts/model_budget_dgx.sh codellama
#SBATCH --job-name=vector-models
#SBATCH --partition=prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300000
#SBATCH --time=24:00:00
#SBATCH --output=logs/vector_models_%j.out
set -euo pipefail
MODEL="${1:?model: qwen or codellama}"
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "Slurm job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}; model ${MODEL}"
git log -1 --oneline
python -m unittest unit_tests.test_net_budget unit_tests.test_vector_study
if [[ "${2:-full}" == "pilot" ]]; then
  python -u compression_topics/vector/scripts/model_budget_study.py --model "$MODEL" --only pilot
else
  python -u compression_topics/vector/scripts/model_budget_study.py --model "$MODEL"
fi
