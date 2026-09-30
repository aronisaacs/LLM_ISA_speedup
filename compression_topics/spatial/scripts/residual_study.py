#!/usr/bin/env python3
"""WikiText sweep, greedy budgets, and GSM8K for adjacent-pair pooling.

Three configs, each with its own sweep and its own greedy fill:
  pair_pool          every included layer fully pools
  residual_no_rope   residual at 25/50/75/100, values and keys unaligned
  residual_rope      the same rungs, keys aligned by one RoPE step

Budgets stop at 50%. That is the most this family can remove: every layer
fully pooled. Scores are read from results.json. A finished simulation is a
row there. A rerun skips a simulation the index already lists.

  python compression_topics/spatial/scripts/residual_study.py --which pair_pool --through sweep
  python compression_topics/spatial/scripts/residual_study.py
  python compression_topics/spatial/scripts/residual_study.py --which residual_rope --from tasks
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

from catalog.compressions import pair_pool, residual_pool  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import GSM8K_20PCT, WIKITEXT_FULL  # noqa: E402
from compression_topics.vector.scripts.plot_vector_study import write_sweep_svg  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, selections_for_budgets, write_budget_run  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
STAGES = ("sweep", "greedy", "tasks", "plots")
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
CONFIGS = {
    "pair_pool": {
        "method": "pair_pool",
        "rope": None,
        "method_kv": pair_pool(),
        "tag": "pair_pool",
        "title": "pair pool",
    },
    "residual_no_rope": {
        "method": "residual_pool",
        "rope": False,
        "method_kv": residual_pool(prune_pct=25, rope=False),
        "tag": "residual_nr",
        "title": "residual pool, no rope",
    },
    "residual_rope": {
        "method": "residual_pool",
        "rope": True,
        "method_kv": residual_pool(prune_pct=25, rope=True),
        "tag": "residual_rope",
        "title": "residual pool, rope",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", default="all", choices=[*CONFIGS, "all"])
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="plots", choices=STAGES)
    args = parser.parse_args()
    start = STAGES.index(args.start)
    end = STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    names = list(CONFIGS) if args.which == "all" else [args.which]
    for name in names:
        for stage in STAGES[start : end + 1]:
            _run_stage(name, stage)


def _run_stage(name: str, stage: str) -> None:
    config = CONFIGS[name]
    figures = FIGURES / name
    if stage == "sweep":
        _multi_run(_sweep_run(config, figures))
        _plot(config, figures)
        return
    if stage == "greedy":
        run_path = write_budget_run(
            _payloads(config),
            figures / "budgets_run.json",
            tasks=(GSM8K_20PCT,),
            results_dir=str(figures),
        )
        print(f"wrote  {run_path}")
        return
    if stage == "tasks":
        _multi_run(json.loads((figures / "budgets_run.json").read_text()))
        return
    _plot(config, figures)


def _sweep_run(config: dict, figures: Path) -> dict:
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=config["method_kv"],
        method_tag=config["tag"],
        results_dir=str(figures),
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


def _payloads(config: dict) -> list[dict]:
    dense_ppl, rows = _scores(config)
    payloads = selections_for_budgets(
        rows,
        LLAMA31_LAYERS,
        config["method_kv"],
        dense_ppl,
        budgets=BUDGETS,
    )
    for payload in payloads:
        payload["tag"] = f"{config['tag']}_{payload['tag']}"
    return payloads


def _scores(config: dict):
    dense_ppl, rows = load_sweep_scores(method=config["method"], pretrained=PRETRAINED)
    return dense_ppl, _kept_rows(rows, config["rope"])


def _kept_rows(rows, rope: bool | None):
    """Keep one residual setting. Both settings share the method name."""
    if rope is None:
        return rows
    kept = []
    for row in rows:
        pipeline = ((row.kv or {}).get("pipeline") or [{}])
        if pipeline[0].get("rope", False) is rope:
            kept.append(row)
    return kept


def _plot(config: dict, figures: Path) -> None:
    dense_ppl, rows = _scores(config)
    if not rows:
        raise ValueError(f"no {config['method']} sweep scores to plot")
    n_layers = max(row.slot.layer for row in rows) + 1
    figures.mkdir(parents=True, exist_ok=True)
    for level in sorted({row.level for row in rows}):
        path = figures / f"sweep_p{level}.svg"
        write_sweep_svg(
            dense_ppl,
            rows,
            level,
            n_layers,
            path,
            title=(
                "Per layer resiliency run on Llama 3.1 8B Instruct using WikiText "
                f"with {config['title']} ({level}%)"
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
