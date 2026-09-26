#!/usr/bin/env python3
"""WikiText sweep curves and task tables for the QJL bit-width study.

  python scripts/plot_qjl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from scripts.plot_vector_study import (  # noqa: E402
    _format_percent,
    _primary_score,
    _table_svg,
    write_sweep_svg,
)

FIGURES = "figures"
_PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
_BITS = (4, 3, 2, 1)
_TASKS = (
    ("ceval-valid", "CEval", "ceval"),
    ("gsm8k", "GSM8K", "gsm8k"),
    ("humaneval_instruct", "HumanEval", "humaneval"),
)


def write_qjl_study(figures_dir=FIGURES, pretrained=_PRETRAINED) -> list[Path]:
    """One WikiText curve per bit width, then CEval, GSM8K, and HumanEval."""
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dense_ppl, rows = load_sweep_scores(method="qjl", pretrained=pretrained)
    n_layers = max(row.slot.layer for row in rows) + 1
    paths = []
    for bits in _BITS:
        path = destination / f"qjl_sweep_{bits}bit.svg"
        write_sweep_svg(
            dense_ppl,
            rows,
            bits,
            n_layers,
            path,
            title=(
                "Per layer resiliency run on Llama 3.1 8B Instruct using WikiText "
                f"with QJL at {bits} bits"
            ),
        )
        paths.append(path)
    for task, title, stem in _TASKS:
        path = destination / f"qjl_{stem}.svg"
        path.write_text(_table_svg(f"{title}, QJL", _task_rows(task, pretrained)))
        paths.append(path)
    return paths


def _task_rows(task: str, pretrained: str) -> list[tuple[str, str, str, str]]:
    from engine.eval_runner.index import simulations

    baseline = None
    points = []
    for record in simulations():
        identity = record.get("identity") or {}
        if identity.get("pretrained") != pretrained or task not in (identity.get("tasks") or []):
            continue
        _name, accuracy = _primary_score({"results": record.get("scores") or {}})
        if accuracy is None:
            continue
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        if not pipeline:
            if baseline is None or record.get("budget") in (0, 0.0):
                baseline = float(accuracy)
            continue
        if pipeline[0].get("method") != "qjl" or "budget" not in record:
            continue
        points.append((float(record["budget"]), float(record.get("compression") or 0.0), float(accuracy)))
    if baseline is None:
        raise ValueError(f"no dense Llama 3.1 {task} baseline in the index")
    points.sort()
    rows = [("Baseline", _show_accuracy(baseline), "0%", "0%")]
    for budget, compression, accuracy in points:
        drop = 0.0 if baseline == 0 else (baseline - accuracy) / baseline * 100.0
        rows.append((_format_percent(budget), _show_accuracy(accuracy), f"{drop:.2f}%", _format_percent(compression)))
    return rows


def _show_accuracy(accuracy: float) -> str:
    shown = accuracy * 100.0 if accuracy <= 1 else accuracy
    return f"{shown:.2f}"


def main() -> None:
    for path in write_qjl_study():
        print(path)


if __name__ == "__main__":
    main()
