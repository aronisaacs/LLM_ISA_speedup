#!/usr/bin/env python3
"""Run the Llama 3.1 8B checksparse and sparsify studies as one interleaved run.

Stages, in order:
  sweep   WikiText singletons for every key and value: checksparse at 25/50/75%,
          sparsify 4:8 on
  greedy  rank-fill at 10–60% mean compression per method, up to what the
          method can reach (sparsify tops out at 50%)
  tasks   CEval, GSM8K, and HumanEval at each method's assignments
  plots   nothing combined yet

Each stage that evaluates makes one multi_run call over both methods, so
multi_run's per-GPU slices hold work from both. The dense baselines the two
methods share appear once.

Greedy reads sweep scores from the repo-root results.json, so a sweep run
with --results-root does not feed it.

  python compression_topics/cross_method_study.py --through sweep
  python compression_topics/cross_method_study.py
  python compression_topics/cross_method_study.py --from tasks --dry-run
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

from catalog.compressions import checksparse_l1, sparsify_nm  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.layer_select.budgets import BUDGETS, LLAMA31_LAYERS, budget_run, write_selections  # noqa: E402
from engine.layer_select.rungs import rungs_for  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

TOPICS = Path(__file__).resolve().parent
# method name, tag, 25% template, figures directory
METHODS = (
    ("checksparse_l1", "checksparse", checksparse_l1(prune_pct=25), TOPICS / "checksparse" / "figures"),
    # 4:8 on or off per slot, so its budgets stop at 50%.
    ("sparsify_nm", "sparsify", sparsify_nm(), TOPICS / "sparsify" / "figures"),
)
BUDGETS_RUN = TOPICS / "cross_method_budgets_run.json"

STAGES = ("sweep", "greedy", "tasks", "plots")
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
# Fields that only label or file a configuration. The rest decide what runs.
_LABELS = {"name", "output_path", "metadata", "layer_slot", "compression_level"}


def sweep_run() -> dict:
    """Dense, then each key and each value at 25%, 50%, and 75%, for each method."""
    configurations = []
    for _method, tag, method_kv, figures in METHODS:
        figures.mkdir(parents=True, exist_ok=True)
        configurations += expand_singleton_configs(
            n_layers=LLAMA31_LAYERS,
            method_kv=method_kv,
            method_tag=tag,
            results_dir=_rel(figures),
            name_prefix="llama31",
            extra=WIKITEXT_FULL,
        )
    return _run(configurations)


def write_budgets_run(path: Path = BUDGETS_RUN) -> Path:
    """Rank-fill each method on its own sweep scores, then write one task run."""
    configurations = []
    for method, _tag, _method_kv, figures in METHODS:
        scored = load_sweep_scores(method=method, pretrained=PRETRAINED)
        selections = write_selections(None, figures, budgets=reachable_budgets(method), scored=scored)
        configurations += budget_run(selections, results_dir=_rel(figures))["configurations"]
        print(f"wrote  {figures}  ({method}, {len(selections)} budgets)")
    path.write_text(json.dumps(_run(configurations), indent=2) + "\n")
    return path


def reachable_budgets(method: str) -> tuple[float, ...]:
    """BUDGETS up to the most a method removes with every slot at its top rung."""
    most = max(rung.fraction for rung in rungs_for(method))
    return tuple(budget for budget in BUDGETS if budget <= most + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="plots", choices=STAGES)
    parser.add_argument("--results-root", type=Path, default=None, metavar="PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    start = STAGES.index(args.start)
    end = STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    for stage in STAGES[start : end + 1]:
        _run_stage(stage, args.results_root, args.dry_run)


def _run_stage(stage: str, results_root: Path | None, dry_run: bool) -> None:
    if stage == "sweep":
        _multi_run(sweep_run(), results_root, dry_run)
        return
    if stage == "greedy":
        print(f"wrote  {write_budgets_run()}")
        return
    if stage == "tasks":
        _multi_run_path(BUDGETS_RUN, results_root, dry_run)
        return
    print("plots  no combined figures yet")


def _run(configurations: list[dict]) -> dict:
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": _unique(configurations),
    }


def _unique(configurations: list[dict]) -> list[dict]:
    """Drop a dense configuration that repeats an earlier one (both methods list it)."""
    seen = set()
    kept = []
    for configuration in configurations:
        if not (configuration.get("kv") or {}).get("pipeline"):
            key = json.dumps(
                {k: v for k, v in configuration.items() if k not in _LABELS}, sort_keys=True
            )
            if key in seen:
                continue
            seen.add(key)
        kept.append(configuration)
    return kept


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
