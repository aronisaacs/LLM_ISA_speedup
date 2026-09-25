#!/usr/bin/env python3
"""Run the Llama 3.1 8B vector-compression study, then write its figures.

Stages, in order:
  sweep   WikiText singletons at 25/50/75% for every key and value
  greedy  rank-fill at 10–60% mean compression
  tasks   CEval, GSM8K, and HumanEval at each of those assignments
  plots   three resiliency curves and one degradation table per task

Scores are read from results/index.json. A finished simulation is a row there.
A rerun skips a simulation the index already lists.

  python scripts/vector_study.py --through sweep
  python scripts/vector_study.py
  python scripts/vector_study.py --from tasks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress  # noqa: E402
from catalog.models import LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, RESULTS, write_budget_run, write_selections  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402
from scripts.plot_vector_study import write_sweep_plots, write_task_tables  # noqa: E402

STAGES = ("sweep", "greedy", "tasks", "plots")
FIGURES = f"{RESULTS}/figures"
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"


def sweep_run() -> dict:
    """Dense, then each key and each value at 25%, 50%, and 75%."""
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=vector_compress(prune_pct=25),
        method_tag="vector",
        results_dir=RESULTS,
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return {
        "model": "hf",
        "batch_size": "auto:4",
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
        for path in write_sweep_plots(out_dir=FIGURES, method="vector_compress", pretrained=PRETRAINED):
            print(f"wrote  {path}")
        return
    if stage == "greedy":
        scored = load_sweep_scores(method="vector_compress", pretrained=PRETRAINED)
        selections = write_selections(None, RESULTS, scored=scored)
        run_path = write_budget_run(selections, Path(RESULTS) / "budgets_run.json")
        print(f"wrote  {run_path}  ({len(selections)} budgets)")
        return
    if stage == "tasks":
        _multi_run(json.loads((Path(RESULTS) / "budgets_run.json").read_text()))
        return
    for path in write_sweep_plots(out_dir=FIGURES, method="vector_compress", pretrained=PRETRAINED):
        print(f"wrote  {path}")
    for path in write_task_tables(out_dir=FIGURES):
        print(f"wrote  {path}")


def _multi_run(run: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run(_multi_run_command(path), cwd=ROOT, check=True)
    finally:
        path.unlink(missing_ok=True)


def _multi_run_command(run_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "engine" / "multi_run.py"),
        "--run",
        str(run_path),
        "--skip-existing",
    ]


if __name__ == "__main__":
    main()
