#!/usr/bin/env python3
"""WikiText sweep, greedy budgets, and GSM8K for each spatial method.

Each method is scored on every key and every value, then rank-filled to
15/30/45/60 percent mean compression. GSM8K is the 20 percent subset.
No figures are written.

  python scripts/spatial_study.py
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

from catalog.compressions import spatial_feature, spatial_pool, spatial_tile, spatial_top1  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import GSM8K_20PCT, WIKITEXT_FULL  # noqa: E402
from engine.eval_runner.progress import mirror_terminal  # noqa: E402
from engine.layer_select.budgets import FIGURES, LLAMA31_LAYERS, budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
BUDGETS = (0.15, 0.30, 0.45, 0.60)
METHODS = (
    ("spatial_pool", spatial_pool),
    ("spatial_top1", spatial_top1),
    ("spatial_tile", spatial_tile),
    ("spatial_feature", spatial_feature),
)


def sweep_run() -> dict:
    """Dense once, then each method on every key and every value."""
    configurations = []
    for tag, builder in METHODS:
        for configuration in expand_singleton_configs(
            n_layers=LLAMA31_LAYERS,
            method_kv=builder(),
            method_tag=tag,
            results_dir=FIGURES,
            name_prefix="llama31",
            extra=WIKITEXT_FULL,
        ):
            if configuration["name"] == "llama31_dense" and configurations:
                continue
            configurations.append(configuration)
    return _run_dict(configurations)


def gsm8k_run() -> dict:
    """One GSM8K list: dense, then each method at 15/30/45/60 percent."""
    selections = []
    for tag, builder in METHODS:
        dense_ppl, rows = load_sweep_scores(method=tag, pretrained=PRETRAINED)
        payloads = selections_for_budgets(rows, LLAMA31_LAYERS, builder(), dense_ppl, budgets=BUDGETS)
        for payload in payloads:
            payload["tag"] = f"{tag}_{payload['tag']}"
        selections.extend(payloads)
    return budget_run(selections, tasks=(GSM8K_20PCT,))


def main() -> None:
    mirror_terminal(ROOT)
    _multi_run(sweep_run())
    _multi_run(gsm8k_run())


def _run_dict(configurations: list[dict]) -> dict:
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


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
