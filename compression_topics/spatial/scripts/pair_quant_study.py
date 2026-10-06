#!/usr/bin/env python3
"""WikiText sweep, greedy budgets, and GSM8K for pair pooling with a quantized residual.

Every pair of tokens is pooled to its mean, and the half-difference is kept
at 8, 4, 2 or 1 bits. A slot (one layer's keys or one layer's values) climbs
8, 4, 2, 1 bits, then merge-only, which stores no residual. Merge-only is
level 100 in the index, because level 0 already means uncompressed.

Budgets stop at 50%. That is the most this family can remove: every slot
merge-only. Scores are read from results.json. A finished simulation is a row
there. A rerun skips a simulation the index already lists. The task is GSM8K.

  python compression_topics/spatial/scripts/pair_quant_study.py --through sweep
  python compression_topics/spatial/scripts/pair_quant_study.py
  python compression_topics/spatial/scripts/pair_quant_study.py --from tasks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import pair_quant  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import GSM8K_20PCT, WIKITEXT_FULL  # noqa: E402
from compression_topics.vector.scripts.plot_vector_study import write_sweep_svg  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, selections_for_budgets, write_budget_run  # noqa: E402
from engine.layer_select.rungs import MERGE_ONLY  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_quant"
STAGES = ("sweep", "greedy", "tasks", "plots")
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
METHOD = "pair_quant"
TAG = "pair_quant"
METHOD_KV = pair_quant(bits=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="plots", choices=STAGES)
    args = parser.parse_args()
    start = STAGES.index(args.start)
    end = STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    for stage in STAGES[start : end + 1]:
        _run_stage(stage)


def _run_stage(stage: str) -> None:
    if stage == "sweep":
        _multi_run(_sweep_run())
        _plot()
        return
    if stage == "greedy":
        run_path = write_budget_run(
            _payloads(),
            FIGURES / "budgets_run.json",
            tasks=(GSM8K_20PCT,),
            results_dir=str(FIGURES),
        )
        print(f"wrote  {run_path}")
        return
    if stage == "tasks":
        _multi_run(json.loads((FIGURES / "budgets_run.json").read_text()))
        return
    _plot()


def _sweep_run() -> dict:
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=METHOD_KV,
        method_tag=TAG,
        results_dir=str(FIGURES),
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


def _payloads() -> list[dict]:
    dense_ppl, rows = load_sweep_scores(method=METHOD, pretrained=PRETRAINED)
    payloads = selections_for_budgets(
        rows,
        LLAMA31_LAYERS,
        METHOD_KV,
        dense_ppl,
        budgets=BUDGETS,
    )
    for payload in payloads:
        payload["tag"] = f"{TAG}_{payload['tag']}"
    return payloads


def _plot() -> None:
    dense_ppl, rows = load_sweep_scores(method=METHOD, pretrained=PRETRAINED)
    if not rows:
        raise ValueError(f"no {METHOD} sweep scores to plot")
    n_layers = max(row.slot.layer for row in rows) + 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    for level in sorted({row.level for row in rows}):
        label = "merge only" if level == MERGE_ONLY else f"{level}-bit residual"
        path = FIGURES / f"sweep_p{level}.svg"
        write_sweep_svg(
            dense_ppl,
            rows,
            level,
            n_layers,
            path,
            title=(
                "Per layer resiliency run on Llama 3.1 8B Instruct using WikiText "
                f"with pair pooling and a quantized residual ({label})"
            ),
        )
        print(f"wrote  {path}")


def _multi_run(run: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "engine" / "multi_run.py"),
                "--run",
                str(path),
                "--skip-existing",
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
