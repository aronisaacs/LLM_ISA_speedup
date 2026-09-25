#!/usr/bin/env python3
"""Run the Llama 3.1 8B vector-compression study, then write its figures.

Stages, in order:
  sweep   WikiText singletons at 25/50/75% for every key and value
  greedy  rank-fill at 10–60% mean compression
  tasks   CEval, GSM8K, and HumanEval at each of those assignments
  plots   three resiliency curves and one degradation table per task

The sweep stage also writes the three curves, so you can stop there.
A rerun skips configurations whose result JSON already has scores.

  python scripts/vector_study.py --through sweep
  python scripts/vector_study.py
  python scripts/vector_study.py --from tasks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.layer_select.budgets import (  # noqa: E402
    BUDGET_RESULTS,
    FIGURES_DIR,
    JSON_DIR,
    SWEEP_RESULTS,
    write_budget_run,
    write_selections,
)
from scripts.plot_vector_study import write_sweep_plots, write_task_tables  # noqa: E402

STAGES = ("sweep", "greedy", "tasks", "plots")


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
        _multi_run(ROOT / "runs" / "vector_sweep.py")
        for path in write_sweep_plots(SWEEP_RESULTS, FIGURES_DIR):
            print(f"wrote  {path}")
        return
    if stage == "greedy":
        selections = write_selections(SWEEP_RESULTS, JSON_DIR)
        run_path = write_budget_run(selections, Path(JSON_DIR) / "budgets_run.json")
        print(f"wrote  {run_path}  ({len(selections)} budgets)")
        return
    if stage == "tasks":
        _multi_run(Path(JSON_DIR) / "budgets_run.json")
        return
    for path in write_sweep_plots(SWEEP_RESULTS, FIGURES_DIR):
        print(f"wrote  {path}")
    for path in write_task_tables(BUDGET_RESULTS, JSON_DIR, FIGURES_DIR):
        print(f"wrote  {path}")


def _multi_run(run_path: Path) -> None:
    subprocess.run(_multi_run_command(run_path), cwd=ROOT, check=True)


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
