#!/usr/bin/env python3
"""WikiText sweep, greedy bit-width climb, then CEval, GSM8K, and HumanEval.

Each key and each value is scored at 4, 3, 2.5 (2 bits + 64 calibrated outlier
channels at 3) and 2 bits. Rank-fill climbs that ladder one slot at a time from
dense, so one layer's keys can stay dense or at 4 bits while another layer's values
go to 2 bits. Budgets are 70/80/85 percent mean compression (proposed; a slot at
2 bits removes 14/16, so 87.5 percent is the top of the ladder). Signs: one random
sign per RoPE pair. Run names carry ``turboquant_mse_rp`` so they do not overwrite
the earlier 4/3/2/1-bit study (independent signs). No figures.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_study.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import turboquant_mse  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.eval_runner.progress import mirror_terminal  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

FIGURES = str(Path(__file__).resolve().parents[1] / "figures")

PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
BUDGETS = (0.70, 0.80, 0.85)
TAG = "turboquant_mse_rp"


def rope_pair_steps(step: dict) -> bool:
    """Sweep rows of this study only (not the earlier independent-sign sweep)."""
    return step.get("signs") == "rope_pairs"


def sweep_run() -> dict:
    """Dense once, then every key and every value at 4, 3, 2.5 and 2 bits."""
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=turboquant_mse(),
        method_tag=TAG,
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
    """Dense, then the greedy assignment at each budget, on each task."""
    dense_ppl, rows = load_sweep_scores(method="turboquant_mse", pretrained=PRETRAINED, step_filter=rope_pair_steps)
    payloads = selections_for_budgets(rows, LLAMA31_LAYERS, turboquant_mse(), dense_ppl, budgets=BUDGETS)
    for payload in payloads:
        payload["tag"] = f"{TAG}_{payload['tag']}"
    return budget_run(payloads, log_samples=True)  # per-question scores for the joint-KL comparison


def main() -> None:
    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table

    mirror_terminal(ROOT)
    ensure_table()
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
