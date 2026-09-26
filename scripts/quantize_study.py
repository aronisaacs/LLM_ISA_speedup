#!/usr/bin/env python3
"""WikiText sweep, greedy bit-width climb, then CEval, GSM8K, and HumanEval.

Each key and each value is scored at int8 and int4. Rank-fill climbs that
ladder one slot at a time, so one layer's keys can stay at 8 bits while
another layer's values go to 4 bits. Budgets are 15/30/45/60/75 percent mean
compression. int4 removes 3/4 of a slot, so 75 percent turns every slot to
4 bits. No figures.

  python scripts/quantize_study.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import quantize  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.eval_runner.progress import mirror_terminal  # noqa: E402
from engine.layer_select.budgets import FIGURES, LLAMA31_LAYERS, budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
BUDGETS = (0.15, 0.30, 0.45, 0.60, 0.75)


def sweep_run() -> dict:
    """Dense once, then every key and every value at 8 bits and at 4 bits."""
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=quantize(),
        method_tag="quantize",
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


def tasks_run() -> dict:
    """Dense, then the greedy assignment at 15/30/45/60/75 percent, on each task."""
    dense_ppl, rows = load_sweep_scores(method="quantize", pretrained=PRETRAINED)
    payloads = selections_for_budgets(rows, LLAMA31_LAYERS, quantize(), dense_ppl, budgets=BUDGETS)
    for payload in payloads:
        payload["tag"] = f"quantize_{payload['tag']}"
    return budget_run(payloads)


def main() -> None:
    mirror_terminal(ROOT)
    _multi_run(sweep_run())
    _multi_run(tasks_run())


def _multi_run(run: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"],
            cwd=ROOT,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
