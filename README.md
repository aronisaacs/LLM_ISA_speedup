# LLM_ISA_speedup

KV-cache compression studies for Llama 3.1 8B. Each experiment is its own script under `compression_topics/`. Scores land in `results.json`. A finished row is skipped on the next run.

## Spatial research

The clean spatial studies and current method terminology are documented in
[compression_topics/spatial/README.md](compression_topics/spatial/README.md).
Use its study-specific Slurm launcher for the current comparison.

## DGX

Lab notes on the machine: `/home/aroni/README.md` (user `aroni`, host `DGX-host-01`).

| | |
|---|---|
| Repo | `/data/users/aroni/projects/LLM_ISA_speedup` |
| Env | `llm_isa_dgx` |
| Conda | `source /opt/conda/etc/profile.d/conda.sh` |
| Model cache | `HF_HUB_CACHE=$HOME/models/huggingface` |
| Dataset cache | `HF_DATASETS_CACHE=$HOME/datasets/huggingface` |
| Scratch | `$HOME/scratch` (not backed up) |

`gpuf` and `gpu` call scripts that are not installed. Use `gpuwho` and `nvidia-smi`. Do not use `gpuall`.

`engine/multi_run.py` starts one process per id in `CUDA_VISIBLE_DEVICES`. A study script that calls it can take several free GPUs. A script that keeps one model loaded and loops uses one GPU. Take only cards `gpuwho` shows as free, and not every card on the machine.

Long work goes through `job`, which is tmux. `tlist` lists sessions. `tat <job>` shows the run. Ctrl+B then D leaves it running. `tkill <job>` stops one. Ctrl+C inside `tat` stops the run.

First-time env setup is in `env/environment.dgx.yml`.

### DGX with slurm

The node has 8 GPUs and one partition, `prod`, with no time limit and no account. It is shared, so ask for only the cards a study needs. Slurm sets `CUDA_VISIBLE_DEVICES` to the granted cards, and `engine/multi_run.py` starts one worker on each. About 32 CPUs per GPU is a fair share.

```bash
cd /data/users/aroni/projects/LLM_ISA_speedup
git pull
mkdir -p logs

cat > logs/study.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=study
#SBATCH --partition=prod
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=96
#SBATCH --mem=300000
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/study_%j.out
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
cd /data/users/aroni/projects/LLM_ISA_speedup
export HF_HUB_CACHE=$HOME/models/huggingface
export HF_DATASETS_CACHE=$HOME/datasets/huggingface
echo "GPUs: $CUDA_VISIBLE_DEVICES"
python -u compression_topics/spatial/scripts/pair_quant_study.py
EOF

sbatch logs/study.sbatch
```

Change `--gres=gpu:N`, the CPU count, and the last line for another study. `squeue -u $USER` shows `PD` (pending, reason in the last column) or `R` (running). `tail -f logs/study_<jobid>.out` follows the log, and the first line lists the granted GPUs. `scancel <jobid>` stops the job. A rerun skips runs already in `results.json`, so submitting again resumes.


A shell for an experiment script looks like this. `git log -1` must be the commit that contains that script. If `tlist` already shows the session name, `tkill` it first. The `python -u` line is that script, not a generic runner.

```bash
cd /data/users/aroni/projects/LLM_ISA_speedup
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
git pull
git log -1 --oneline

gpuwho
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tlist

job vector_study 'bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate llm_isa_dgx && cd /data/users/aroni/projects/LLM_ISA_speedup && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export HF_HUB_CACHE=$HOME/models/huggingface && export HF_DATASETS_CACHE=$HOME/datasets/huggingface && python -u compression_topics/vector/scripts/vector_study.py"'
```

See it with `tat vector_study`. Ctrl+B then D detaches.

## Extending the code

Compression kernels live under `compression_topics/<topic>/algorithms`. Register
an `apply(tensor, *, layer_idx, target, ...)` function in `METHODS`; declare its
configuration options explicitly in the signature. The parser rejects unknown
options before loading a model. A forwarding wrapper can register its option
signature in `OPTION_SIGNATURES`.

Methods that finish groups across decode updates can register an optional
`AFTER_APPEND` callback. It receives the full cached tensor, target, layer,
absolute `start`/`end`, RoPE tables, and method options. The method owns group
boundaries and reconstruction; the shared cache owns dispatch. A new method
normally needs no edits to the cache or evaluator.

`engine/kv_compress/metrics.py` owns evaluation statistics. Methods with a byte
model report per-slot `dense_bits` and `stored_bits`; the cache observes dense
bytes for all layers and both targets. Storage measurements describe the
simulated representation at 16 bits per dense value, not allocated tensor
memory. Unsupported methods or overlapping representations leave measured
storage unavailable rather than claiming zero overhead. Spatial storage formulas
are defined in `compression_topics/spatial/algorithms/storage.py`.

`engine/layer_select/calibration.py` owns screening, matched-chunk comparisons,
refinement, and selection. Studies define candidates and their grouping, final
tasks, and summaries. Fixed-rung and calibrated studies use the same greedy
allocator. `engine/eval_runner/chunks.py` runs chunk-loss evaluations using the
shared model loader and worker lifecycle; the old spatial runner path remains a
small compatibility entry point. Spatial planners accept `--model-args`,
`--layers`, and `--head-dim` for offline planning and validate the planned shape
against the loaded model before evaluation. Model defaults remain in the study.

## Results and resuming

`results.json` is the authoritative compact score ledger. Updates are locked
across workers and written atomically. Corrupt ledgers cause an error and are
never silently replaced. Run identities include the model options, compression
pipeline, task settings, prompt settings, and nondefault evaluation seeds.
Execution settings such as device do not prevent score reuse.

New budget rows distinguish `budget`, `planned_compression`, and, when measured,
`measured_compression` with a `compression_target` (`k`, `v`, or `kv`). The
`storage` field contains measured K, V, and combined totals. Historical
`compression` fields remain unchanged and should be treated as planned savings.

Calibration chunks, candidate manifests, and detailed per-configuration results
stay in their study directories: they are intermediate data used for resuming and
paired comparisons. Final spatial task scores go into the root ledger. Detailed
JSON files preserve the historical `gate_stats` field name for compatibility,
although collection now belongs to the shared metrics module. Reports and charts
are derived artifacts. Study scripts remain plan-only unless `--execute` is given.
