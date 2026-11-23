#!/usr/bin/env bash
# Vector and spatial compression on Llama, Qwen and CodeLlama: isolated greedy on 4-domain KL, then tasks.
#   sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh                       # everything
#   sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh --through greedy      # KL and maps only
#   sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh --models qwen --methods spatial
#   sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh --timing 4 --models llama   # quick timing check
#   sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh --from greedy        # added budgets, on the saved KL scores
# Resubmitting resumes (saved KL scores and results.json rows are skipped).
#SBATCH --job-name=kl-greedy
#SBATCH --partition=prod
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=300000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/kl_greedy_%j.out
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd "$HOME/projects/LLM_ISA_speedup"
export HF_HUB_CACHE="$HOME/models/huggingface"
export HF_DATASETS_CACHE="$HOME/datasets/huggingface"
export OMP_NUM_THREADS=2  # one process per GPU, mostly GPU-bound: 8 CPUs for 4 GPUs (Aron, 2026-11-23)
export PYTHONUNBUFFERED=1
echo "Slurm job ${SLURM_JOB_ID:-manual}; GPUs ${CUDA_VISIBLE_DEVICES:-unset}; args: $*"
git log -1 --oneline
python -m unittest unit_tests.test_kl_greedy_rerun unit_tests.test_spatial_batching unit_tests.test_joint_kl unit_tests.test_spatial_algorithms unit_tests.test_storage_contract
python -u compression_topics/kl_greedy/scripts/rerun.py "$@"
