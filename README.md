# LLM_ISA_speedup

KV-cache compression studies for Llama 3.1 8B. Each experiment is its own script under `compression_topics/`. Scores land in `results.json`. A finished row is skipped on the next run.

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
