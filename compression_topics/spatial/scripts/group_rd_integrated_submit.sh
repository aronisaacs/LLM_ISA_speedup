#!/usr/bin/env bash
# Run from the DGX checkout: bash compression_topics/spatial/scripts/group_rd_integrated_submit.sh
# Calibration uses six GPUs; validation/C-Eval have dense + five saving levels.
# The second job starts only after successful calibration, never in parallel.
set -euo pipefail
cd /data/users/aroni/projects/LLM_ISA_speedup
mkdir -p logs
launcher=compression_topics/spatial/scripts/group_rd_integrated_dgx.sh
calibration_job=$(sbatch --parsable "$launcher" "$@" --through allocate)
evaluation_job=$(sbatch --parsable --dependency="afterok:${calibration_job%%;*}" --kill-on-invalid-dep=yes \
    --gres=gpu:6 --cpus-per-task=192 --mem=600000 \
    --job-name=spatial-rd-final "$launcher" "$@" --from validate --through summary)
echo "Calibration: $calibration_job (6 GPUs, pilot then fresh layer calibration)"
echo "Evaluation: $evaluation_job (6 GPUs, dense plus 10/20/30/40/50%, waits for calibration)"
echo "Logs: logs/spatial_rd_integrated_<jobid>.out"
echo "Preliminary results: compression_topics/spatial/figures/group_rd_integrated/*/*.partial.json"
