# LLM_ISA_speedup

KV-cache compression studies for Llama 3.1 8B. Scores land in `results.json`. A finished row is skipped on the next run.

## DGX

Lab notes on the machine: `/home/aroni/README.md` (user `aroni`, host `DGX-host-01`).

| | |
|---|---|
| Repo | `/data/users/aroni/projects/LLM_ISA_speedup` |
| Env | `llm_isa_dgx` |
| Conda | `source /opt/conda/etc/profile.d/conda.sh` |
| Model cache | `HF_HUB_CACHE=$HOME/models/huggingface` |
| Dataset cache | `HF_DATASETS_CACHE=$HOME/datasets/huggingface` |
| Job logs | `$HOME/logs/<job>.log` |
| Scratch | `$HOME/scratch` (not backed up) |

`gpuf` and `gpu` call scripts that are not installed. Use `gpuwho` and `nvidia-smi`. Do not use `gpuall`.

`engine/multi_run.py` starts one process per id in `CUDA_VISIBLE_DEVICES`. A study script that calls it can take several free GPUs. A script that keeps one model loaded and loops uses one GPU. Take only cards `gpuwho` shows as free, and not every card on the machine.

Long work goes through `job`, which is tmux. `tlist` lists sessions, `tat <job>` attaches, Ctrl+B then D detaches, `tkill <job>` stops one. Ctrl+C inside `tat` stops the run. Ctrl+C on `tail -f` only stops the viewer.

First-time env setup is in `env/environment.dgx.yml`.

## Launch script

Fill in `GPUS`, `JOB`, `COMMIT`, and `COMMAND`. `git log -1` must start with `COMMIT` before `job` runs. If `tlist` already shows `JOB`, run `tkill` on it first.

```bash
cd /data/users/aroni/projects/LLM_ISA_speedup
source /opt/conda/etc/profile.d/conda.sh
conda activate llm_isa_dgx
git pull
git log -1 --oneline

gpuwho
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tlist

GPUS=0,1,2,3
JOB=study
COMMIT=0123456
COMMAND="python -u compression_topics/vector/scripts/vector_study.py"

mkdir -p "$HOME/logs"
job "$JOB" 'bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate llm_isa_dgx && cd /data/users/aroni/projects/LLM_ISA_speedup && export CUDA_VISIBLE_DEVICES='"$GPUS"' && export HF_HUB_CACHE=$HOME/models/huggingface && export HF_DATASETS_CACHE=$HOME/datasets/huggingface && '"$COMMAND"' 2>&1 | tee $HOME/logs/'"$JOB"'.log"'
```

Watch with `tail -f ~/logs/$JOB.log`.
