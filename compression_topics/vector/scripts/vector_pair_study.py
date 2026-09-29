#!/usr/bin/env python3
"""WikiText sweep, greedy budgets, and task evals for pair-magnitude vector compression.

Stages, in order:
  sweep   WikiText singletons at 25/50/75% for every key and value
  greedy  rank-fill at 10–60% mean compression
  tasks   CEval, GSM8K, and HumanEval at each of those assignments
  plots   one WikiText resiliency curve per rung

Scores are read from results.json. A finished simulation is a row there.
A rerun skips a simulation the index already lists.

  python compression_topics/vector/scripts/vector_pair_study.py --through sweep
  python compression_topics/vector/scripts/vector_pair_study.py
  python compression_topics/vector/scripts/vector_pair_study.py --from tasks
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

from catalog.compressions import vector_compress_pair  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from compression_topics.vector.scripts.plot_vector_study import write_sweep_plots  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, selections_for_budgets, write_budget_run  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

FIGURES = str(Path(__file__).resolve().parents[1] / "figures" / "pair")
STAGES = ("sweep", "greedy", "tasks", "plots")
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"


def sweep_run() -> dict:
    """Dense, then each key and each value at 25%, 50%, and 75%."""
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=vector_compress_pair(prune_pct=25),
        method_tag="vector_pair",
        results_dir=FIGURES,
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


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
        _multi_run(sweep_run())
        for path in write_sweep_plots(out_dir=FIGURES, method="vector_compress_pair", pretrained=PRETRAINED):
            print(f"wrote  {path}")
        return
    if stage == "greedy":
        run_path = write_budget_run(_payloads(), Path(FIGURES) / "budgets_run.json", results_dir=FIGURES)
        print(f"wrote  {run_path}")
        return
    if stage == "tasks":
        _multi_run(json.loads((Path(FIGURES) / "budgets_run.json").read_text()))
        return
    for path in write_sweep_plots(out_dir=FIGURES, method="vector_compress_pair", pretrained=PRETRAINED):
        print(f"wrote  {path}")


def _payloads() -> list[dict]:
    dense_ppl, rows = load_sweep_scores(method="vector_compress_pair", pretrained=PRETRAINED)
    payloads = selections_for_budgets(
        rows, LLAMA31_LAYERS, vector_compress_pair(prune_pct=25), dense_ppl
    )
    for payload in payloads:
        payload["tag"] = f"vector_pair_{payload['tag']}"
    return payloads


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
