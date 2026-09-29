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
| Job logs | `$HOME/logs/<job>.log` |
| Scratch | `$HOME/scratch` (not backed up) |

`gpuf` and `gpu` call scripts that are not installed. Use `gpuwho` and `nvidia-smi`. Do not use `gpuall`.

`engine/multi_run.py` starts one process per id in `CUDA_VISIBLE_DEVICES`. A study script that calls it can take several free GPUs. A script that keeps one model loaded and loops uses one GPU. Take only cards `gpuwho` shows as free, and not every card on the machine.

Long work goes through `job`, which is tmux. `tlist` lists sessions, `tat <job>` attaches, Ctrl+B then D detaches, `tkill <job>` stops one. Ctrl+C inside `tat` stops the run. Ctrl+C on `tail -f` only stops the viewer.

First-time env setup is in `env/environment.dgx.yml`.

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

mkdir -p "$HOME/logs"
job vector_study 'bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate llm_isa_dgx && cd /data/users/aroni/projects/LLM_ISA_speedup && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export HF_HUB_CACHE=$HOME/models/huggingface && export HF_DATASETS_CACHE=$HOME/datasets/huggingface && python -u compression_topics/vector/scripts/vector_study.py 2>&1 | tee $HOME/logs/vector_study.log"'
```

Watch with `tail -f ~/logs/vector_study.log`.
