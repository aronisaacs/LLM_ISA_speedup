# Vector and spatial compression on all three models: greedy on KL

One protocol for vector and spatial compression on Llama 3.1 8B-Instruct, Qwen3.5-9B and
CodeLlama 7B (decided 2026-11-21). It replaces the per-slot WikiText perplexity sweeps
(`vector/scripts/model_budget_study.py`, `spatial/scripts/ladder.py`): single-slot perplexity
changes were within noise (Llama, one slot at 25 %: 8.81-8.84 around dense 8.83).

1. **Calibration set**, one per model: 512 examples, 128 each of WikiText, ARC 5-shot, GSM8K and
   MBPP, scored on the dense model's own greedy answers (`engine/layer_select/joint_kl`). Llama:
   `catalog/calibration/llama31_8b_mixed4` (E18). Qwen (chat, thinking off, 5-shot GSM8K, 3-shot MBPP)
   and CodeLlama (plain prompts, 0-shot GSM8K and MBPP) are built on first run into
   `catalog/calibration/{qwen35_9b,codellama_7b}_mixed4`.
2. **Singles**: KL(dense || compressed) for every KV slot (one layer's K or V) alone at every level,
   and for a uniform reference map. A slot's score is `kl_mean`: per domain KL / reference KL,
   equal-weight mean over the 4 domains.
3. **Greedy** (isolated, not the sequential joint-KL climb): from dense, raise the slot whose next
   level adds the least score per unit of net saving, until the mean net saving over the KV slots
   reaches each budget.
4. **Tasks**: dense and each budget's map. Llama and Qwen: C-Eval 5-shot, GSM8K, HumanEval-instruct.
   CodeLlama: HumanEval with its body-indent prompt (`humaneval_codellama`).

| | vector | spatial |
|---|---|---|
| method | per-scalar bitmask prune (`vector_compress`) | pairs, query-aware keys, dense/PR0/PR16/PR32 |
| levels per slot | 25 / 50 / 75 % of values | 10 / 25 / 40 / 47 / 49 % saving |
| net of | 1-bit-per-value mask | 2-bit pair tags |
| window | tasks: newest **32 tokens** exact (`recent_window`); calibration: none | **32 tokens** (calibration and tasks) |
| budgets (net) | 10 / 20 / 30 / 40 / 50 / 55 / 60 % | 10 / 20 / 30 / 35 / 40 % |
| reference map | every slot 37 % (30.5 % net) | every slot 30 % |
| task batch size | 8 | 8 for GSM8K and HumanEval, 1 for C-Eval |

**Savings are nominal** (every token compressed). The tokens a window keeps exact are not
subtracted, so the true saving is a few points lower. Spatial's measured storage is 95-98 % of
nominal on these tasks (Llama p40: 38.5 % GSM8K, 38.1 % HumanEval, 39.2 % C-Eval). Vector is not
measured; by token count (~400-1100 tokens per sequence, 32 exact) it is about 92-96 % of nominal.
Footnote for plots: "true savings are within a few points below the plotted values (32-token
exact window)."

Qwen3.5 has a KV cache only in its 8 full-attention layers (3, 7, ..., 31): 16 slots, and budgets
are shares of that cache. Spatial query weights are each model's official calibration
(`spatial/figures/official_<model>/query_weights.pt`).

```bash
python compression_topics/kl_greedy/scripts/preflight.py             # login node, CPU, offline: all inputs present?
python compression_topics/kl_greedy/scripts/preflight.py --download  # fetch missing models and datasets
sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh                  # everything, resumable
sbatch compression_topics/kl_greedy/scripts/rerun_dgx.sh --through greedy # KL scores and maps only
```

Outputs per model and method in `results/<model>/<method>/`: `singles.json` (raw KL per domain of
every candidate), `selected_p<budget>.json` and `selections.json` (maps; `kv` is what the tasks run,
`calibration_kv` what the KL scored), `tasks_run.json`,
`table.txt`. Task scores go to the repo's `results.json` as usual.

The spatial window is the `window` option of `SpatialConfig` (default 32). Spec steps without a
`window` key are 16-token windows: every spatial result recorded before 2026-11-21.

Spatial runs batched generation: each row's windows start at its first real token, read from the
attention mask lm-eval passes to `generate` (`engine/kv_compress/padding.py`), so a row is compressed
exactly as in a batch of one (`unit_tests/test_spatial_batching.py`: same tokens and storage bits).
lm-eval's log-likelihood calls (C-Eval) are right-padded with no mask, so they stay at batch size 1;
spatial refuses a larger batch without a mask.
