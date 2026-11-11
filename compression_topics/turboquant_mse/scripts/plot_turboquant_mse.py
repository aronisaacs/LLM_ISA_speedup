#!/usr/bin/env python3
"""WikiText sweep curves and task tables for the TurboQuant-MSE bit-width study.

  python compression_topics/turboquant_mse/scripts/plot_turboquant_mse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from compression_topics.vector.scripts.plot_vector_study import (  # noqa: E402
    _format_percent,
    _primary_score,
    _table_svg,
    write_sweep_svg,
)

FIGURES = str(Path(__file__).resolve().parents[1] / "figures")
_PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
_BITS = (4, 3, 2, 1)
_TASKS = (
    ("ceval-valid", "CEval", "ceval"),
    ("gsm8k", "GSM8K", "gsm8k"),
    ("humaneval_instruct", "HumanEval", "humaneval"),
)


def write_turboquant_mse_study(figures_dir=FIGURES, pretrained=_PRETRAINED) -> list[Path]:
    """One WikiText curve per bit width, then CEval, GSM8K, and HumanEval."""
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dense_ppl, rows = load_sweep_scores(method="turboquant_mse", pretrained=pretrained)
    n_layers = max(row.slot.layer for row in rows) + 1
    paths = []
    for bits in _BITS:
        path = destination / f"turboquant_mse_sweep_{bits}bit.svg"
        write_sweep_svg(
            dense_ppl,
            rows,
            bits,
            n_layers,
            path,
            title=(
                "Per layer resiliency run on Llama 3.1 8B Instruct using WikiText "
                f"with TurboQuant-MSE at {bits} bits"
            ),
        )
        paths.append(path)
    for task, title, stem in _TASKS:
        path = destination / f"turboquant_mse_{stem}.svg"
        path.write_text(_table_svg(f"{title}, TurboQuant-MSE", _task_rows(task, pretrained)))
        paths.append(path)
    return paths


def _task_rows(task: str, pretrained: str) -> list[tuple[str, str, str, str]]:
    """Baseline row, then one row per budget.

    The baseline is the dense run with the same settings as the compressed runs
    (chat template, few-shot, generation options, dtype), so both sides match.
    """
    from engine.eval_runner.cache import identities_match
    from engine.eval_runner.index import simulations

    dense = []
    points = []
    settings = []
    for record in simulations():
        identity = record.get("identity") or {}
        if identity.get("pretrained") != pretrained or task not in (identity.get("tasks") or []):
            continue
        _name, accuracy = _primary_score({"results": record.get("scores") or {}})
        if accuracy is None:
            continue
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        if not pipeline:
            dense.append((_without_kv(identity), float(accuracy)))
            continue
        if pipeline[0].get("method") != "turboquant_mse" or "budget" not in record:
            continue
        if not any(identities_match(seen, _without_kv(identity)) for seen in settings):
            settings.append(_without_kv(identity))
        points.append((float(record["budget"]), float(record.get("measured_compression", record.get("planned_compression", record.get("compression"))) or 0.0), float(accuracy)))
    if len(settings) != 1:
        raise ValueError(f"expected TurboQuant-MSE {task} runs with one set of settings, found {len(settings)}")
    matches = [accuracy for identity, accuracy in dense if identities_match(identity, settings[0])]
    if not matches:
        raise ValueError(f"no dense Llama 3.1 {task} baseline with settings {settings[0]}")
    baseline = matches[0]
    points.sort()
    rows = [("Baseline", _show_accuracy(baseline), "0%", "0%")]
    for budget, compression, accuracy in points:
        drop = 0.0 if baseline == 0 else (baseline - accuracy) / baseline * 100.0
        rows.append((_format_percent(budget), _show_accuracy(accuracy), f"{drop:.2f}%", _format_percent(compression)))
    return rows


def _without_kv(identity: dict) -> dict:
    """A run's identity without its KV compression: the settings both sides must share."""
    return {key: value for key, value in identity.items() if key != "kv"}


def _show_accuracy(accuracy: float) -> str:
    shown = accuracy * 100.0 if accuracy <= 1 else accuracy
    return f"{shown:.2f}"


def main() -> None:
    for path in write_turboquant_mse_study():
        print(path)


if __name__ == "__main__":
    main()
