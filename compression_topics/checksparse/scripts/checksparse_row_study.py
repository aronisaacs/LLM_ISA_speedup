#!/usr/bin/env python3
"""Llama 3.1 8B checksparse study with Mustafa's row-wide tile choice.

checksparse_row picks the weakest L1 tiles across all KV heads of a token at
once. checksparse_l1 (per head) stays as it was, and its rows in the index are
kept apart by method name.

Stages, in order:
  sweep   WikiText singletons: dense, then every key and value at 25/50/75%
  greedy  rank-fill at 10–60% mean compression over all 64 slots
  tasks   CEval, GSM8K (full set), and HumanEval at each budget, plus dense

Greedy reads sweep scores from the repo-root results.json, so a sweep run
with --results-root does not feed it. Every stage skips rows already scored.

  python compression_topics/checksparse/scripts/checksparse_row_study.py
  python compression_topics/checksparse/scripts/checksparse_row_study.py --through sweep
  python compression_topics/checksparse/scripts/checksparse_row_study.py --from tasks --dry-run
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

from catalog.compressions import checksparse_row  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.layer_select.budgets import BUDGETS, LLAMA31_LAYERS, budget_run, write_selections  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

METHOD = "checksparse_row"
TAG = "checksparse_row"
FIGURES = Path(__file__).resolve().parents[1] / "figures" / "row"
TASKS_RUN = FIGURES / "budgets_run.json"
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
STAGES = ("sweep", "greedy", "tasks")


def sweep_run() -> dict:
    """Dense, then each key and each value at 25%, 50%, and 75%."""
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=checksparse_row(prune_pct=25),
        method_tag=TAG,
        results_dir=_rel(FIGURES),
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return _run(configurations)


def write_tasks_run(path: Path = TASKS_RUN, budgets=BUDGETS) -> Path:
    """Rank-fill on the sweep scores, write the selections, then one task run."""
    scored = load_sweep_scores(method=METHOD, pretrained=PRETRAINED)
    selections = write_selections(None, FIGURES, budgets=budgets, scored=scored)
    for selection in selections:
        selection["tag"] = f"{TAG}_{selection['tag']}"
    run = budget_run(selections, results_dir=_rel(FIGURES))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="tasks", choices=STAGES)
    parser.add_argument("--results-root", type=Path, default=None, metavar="PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    start = STAGES.index(args.start)
    end = STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[start : end + 1]:
        _run_stage(stage, args.results_root, args.dry_run)


def _run_stage(stage: str, results_root: Path | None, dry_run: bool) -> None:
    if stage == "sweep":
        _multi_run(sweep_run(), results_root, dry_run)
        return
    if stage == "greedy":
        print(f"wrote  {write_tasks_run()}")
        return
    _multi_run_path(TASKS_RUN, results_root, dry_run)


def _run(configurations: list[dict]) -> dict:
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _multi_run(run: dict, results_root: Path | None, dry_run: bool) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        _multi_run_path(path, results_root, dry_run)
    finally:
        path.unlink(missing_ok=True)


def _multi_run_path(run_path: Path, results_root: Path | None, dry_run: bool) -> None:
    subprocess.run(_multi_run_command(run_path, results_root, dry_run), cwd=ROOT, check=True)


def _multi_run_command(run_path: Path, results_root: Path | None = None, dry_run: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "engine" / "multi_run.py"),
        "--run",
        str(run_path),
        "--skip-existing",
    ]
    if results_root is not None:
        command += ["--results-root", str(results_root)]
    if dry_run:
        command.append("--dry-run")
    return command


if __name__ == "__main__":
    main()
