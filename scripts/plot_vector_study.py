#!/usr/bin/env python3
"""Sweep resiliency curves and per-task degradation tables for the vector study.

  python scripts/plot_vector_study.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_select.budgets import BUDGET_RESULTS, FIGURES_DIR, JSON_DIR, SWEEP_RESULTS  # noqa: E402
from layer_select.levels import LEVELS  # noqa: E402
from layer_select.scores import load_sweep_scores, word_perplexity  # noqa: E402

TASK_TITLES = {
    "ceval-valid": "CEval",
    "gsm8k": "GSM8K",
    "humaneval_instruct": "HumanEval",
}

_TAG = re.compile(r"^(.*)_p(\d+)$")


def write_sweep_plots(scores_dir=SWEEP_RESULTS, out_dir=FIGURES_DIR) -> list[Path]:
    dense_ppl, rows = load_sweep_scores(scores_dir)
    n_layers = max(row.slot.layer for row in rows) + 1
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for level in LEVELS:
        path = destination / f"sweep_p{level}.svg"
        write_sweep_svg(dense_ppl, rows, level, n_layers, path)
        paths.append(path)
    return paths


def write_sweep_svg(dense_ppl, rows, level, n_layers, output: Path) -> None:
    keys, values = _curves(rows, level, n_layers)
    output.write_text(_sweep_svg(dense_ppl, keys, values, level))


def write_task_tables(results_dir=BUDGET_RESULTS, study_dir=JSON_DIR, out_dir=FIGURES_DIR) -> list[Path]:
    index = _load_budget_index(study_dir)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(results_dir).glob("*.json")):
        point = _task_point(path, index)
        if point is None:
            continue
        grouped[point["task"]].append(point)
    if not grouped:
        raise ValueError(f"no task results in {results_dir}")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, points in grouped.items():
        points.sort(key=lambda item: item["budget"])
        title = TASK_TITLES.get(task, task)
        path = destination / f"{_file_stem(task)}.svg"
        path.write_text(_table_svg(title, _table_rows(points)))
        paths.append(path)
    return paths


def _curves(rows, level, n_layers):
    keys = [None] * n_layers
    values = [None] * n_layers
    for row in rows:
        if row.level != level:
            continue
        series = keys if row.slot.target == "k" else values
        series[row.slot.layer] = row.ppl
    missing = [
        f"{target}{layer}"
        for target, series in (("k", keys), ("v", values))
        for layer, value in enumerate(series)
        if value is None
    ]
    if missing:
        raise ValueError(f"sweep is missing {level}% scores for {missing}")
    return keys, values


def _sweep_svg(dense_ppl, keys, values, level) -> str:
    width, height = 920, 520
    pad_left, pad_right, pad_top, pad_bottom = 64, 28, 58, 56
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    n_layers = len(keys)
    band = dense_ppl * 1.05
    samples = list(keys) + list(values) + [dense_ppl, band]
    low = min(samples)
    high = max(samples)
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08

    def x_at(layer):
        if n_layers == 1:
            return pad_left + plot_w / 2
        return pad_left + plot_w * layer / (n_layers - 1)

    def y_at(value):
        return pad_top + plot_h * (1 - (value - low) / (high - low))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="WikiText perplexity by layer at {level}% vector compression">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad_left}" y="28" font-size="15" font-family="system-ui,sans-serif" fill="#212529">'
        f"Per layer resiliency run on Llama 3.1 8B Instruct using WikiText "
        f"with vector compression ({level}% compression)</text>",
    ]
    band_top = y_at(band)
    parts.append(
        f'<rect x="{pad_left}" y="{band_top:.1f}" width="{plot_w}" height="{y_at(low) - band_top:.1f}" '
        f'fill="#d8f3dc"/>'
    )
    for tick in _ticks(low, high, 5):
        y = y_at(tick)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#495057">{tick:.2f}</text>'
        )
    step = 1 if n_layers <= 16 else 2
    for layer in range(0, n_layers, step):
        x = x_at(layer)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#495057">{layer}</text>'
        )
    baseline_y = y_at(dense_ppl)
    parts.append(
        f'<line x1="{pad_left}" y1="{baseline_y:.1f}" x2="{pad_left + plot_w}" y2="{baseline_y:.1f}" '
        f'stroke="#212529" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    parts.append(_polyline(keys, x_at, y_at, "#1f77b4"))
    parts.append(_polyline(values, x_at, y_at, "#ff7f0e"))
    parts.append(
        f'<text x="{width / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" '
        f'font-family="system-ui,sans-serif" fill="#212529">Layer index</text>'
        f'<text x="16" y="{pad_top + plot_h / 2:.0f}" font-size="12" '
        f'font-family="system-ui,sans-serif" fill="#212529" '
        f'transform="rotate(-90 16 {pad_top + plot_h / 2:.0f})">Perplexity</text>'
    )
    parts.append(_legend(pad_left + plot_w - 168, pad_top + 12))
    parts.append("</svg>")
    return "".join(parts)


def _polyline(series, x_at, y_at, color) -> str:
    points = " ".join(f"{x_at(layer):.1f},{y_at(value):.1f}" for layer, value in enumerate(series))
    dots = "".join(
        f'<circle cx="{x_at(layer):.1f}" cy="{y_at(value):.1f}" r="2.5" fill="{color}"/>'
        for layer, value in enumerate(series)
    )
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{points}"/>' + dots
    )


def _legend(x, y) -> str:
    rows = (
        ("#d8f3dc", "5% degradation band", True),
        ("#1f77b4", "K layers", False),
        ("#ff7f0e", "V layers", False),
        ("#212529", "Baseline", False),
    )
    parts = [
        f'<rect x="{x}" y="{y}" width="176" height="78" fill="#ffffff" stroke="#ced4da"/>'
    ]
    for index, (color, label, band) in enumerate(rows):
        row_y = y + 14 + index * 16
        if band:
            parts.append(f'<rect x="{x + 8}" y="{row_y - 8}" width="18" height="10" fill="{color}" stroke="#adb5bd"/>')
        elif label == "Baseline":
            parts.append(
                f'<line x1="{x + 8}" y1="{row_y - 3}" x2="{x + 26}" y2="{row_y - 3}" '
                f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4 2"/>'
            )
        else:
            parts.append(
                f'<line x1="{x + 8}" y1="{row_y - 3}" x2="{x + 26}" y2="{row_y - 3}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{x + 32}" y="{row_y}" font-size="11" font-family="system-ui,sans-serif" '
            f'fill="#212529">{label}</text>'
        )
    return "".join(parts)


def _table_svg(title, rows) -> str:
    width = 760
    row_h = 36
    header_h = 42
    top = 56
    height = top + header_h + row_h * len(rows) + 16
    columns = (180, 140, 180, 200)
    headers = ("Experiment", "Accuracy ↑", "Degradation", "Overall KV Compression")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{title} degradation by compression budget">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-size="18" font-family="system-ui,sans-serif" font-weight="700" '
        f'fill="#212529">Llama 3.1 8B Instruct — {title}</text>',
    ]
    x = 24
    parts.append(f'<rect x="{x}" y="{top}" width="{sum(columns)}" height="{header_h}" fill="#1b4f72"/>')
    cursor = x
    for header, column in zip(headers, columns):
        parts.append(
            f'<text x="{cursor + 12}" y="{top + 26}" font-size="13" font-family="system-ui,sans-serif" '
            f'font-weight="700" fill="#ffffff">{header}</text>'
        )
        cursor += column
    for index, row in enumerate(rows):
        y = top + header_h + index * row_h
        fill = "#eef2f5" if index % 2 == 0 else "#f7f9fb"
        parts.append(f'<rect x="{x}" y="{y}" width="{sum(columns)}" height="{row_h}" fill="{fill}"/>')
        cursor = x
        for value, column in zip(row, columns):
            parts.append(
                f'<text x="{cursor + 12}" y="{y + 23}" font-size="13" font-family="system-ui,sans-serif" '
                f'fill="#212529">{value}</text>'
            )
            cursor += column
    parts.append("</svg>")
    return "".join(parts)


def _table_rows(points) -> list[tuple[str, str, str, str]]:
    baseline = next(point["accuracy"] for point in points if point["budget"] == 0)
    rows = []
    for point in points:
        if point["budget"] == 0 or baseline == 0:
            degradation = "0%"
        else:
            drop = (baseline - point["accuracy"]) / baseline * 100.0
            degradation = f"{drop:.2f}%"
        label = "Baseline" if point["budget"] == 0 else _format_percent(point["budget"])
        accuracy = point["accuracy"] * 100.0 if point["accuracy"] <= 1 else point["accuracy"]
        rows.append((label, f"{accuracy:.2f}", degradation, _format_percent(point["compression"])))
    return rows


def _task_point(path: Path, index: dict) -> dict | None:
    payload = json.loads(path.read_text())
    task, accuracy = _primary_score(payload)
    if task is None:
        return None
    tag = _tag_from_stem(path.stem)
    if tag is None:
        return None
    entry = index.get(tag)
    if entry is None:
        entry = _entry_from_metadata(payload, tag)
    if entry is None:
        raise ValueError(f"{path} has tag {tag} but no budget entry")
    return {
        "task": task,
        "budget": float(entry["budget"]),
        "compression": float(entry["compression"]),
        "accuracy": accuracy,
    }


def _primary_score(payload: dict) -> tuple[str | None, float | None]:
    grouped = _pick_from_table(payload.get("groups") or {})
    if grouped[0] is not None:
        return grouped
    try:
        perplexity = word_perplexity(payload)
    except ValueError:
        perplexity = None
    if perplexity is not None:
        task = next(iter((payload.get("results") or {})))
        return task, perplexity
    return _pick_from_table(payload.get("results") or {})


def _pick_from_table(table: dict) -> tuple[str | None, float | None]:
    preferred = (
        "acc,none",
        "acc_norm,none",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "pass@1,create_test",
        "pass@1,none",
    )
    for task, metrics in table.items():
        if not isinstance(metrics, dict):
            continue
        for name in preferred:
            if metrics.get(name) is not None:
                return task, float(metrics[name])
        for key, value in metrics.items():
            if key == "alias" or "stderr" in key or not isinstance(value, (int, float)):
                continue
            return task, float(value)
    return None, None


def _load_budget_index(study_dir) -> dict:
    path = Path(study_dir) / "budgets.json"
    if not path.is_file():
        return {}
    return {item["tag"]: item for item in json.loads(path.read_text())}


def _entry_from_metadata(payload, tag) -> dict | None:
    configs = payload.get("configs") or {}
    for task_config in configs.values():
        metadata = (task_config or {}).get("metadata") or {}
        if "kv_budget" in metadata and "kv_compression" in metadata:
            return {"tag": tag, "budget": metadata["kv_budget"], "compression": metadata["kv_compression"]}
    return None


def _tag_from_stem(stem: str) -> str | None:
    if stem.endswith("_dense"):
        return "dense"
    match = _TAG.fullmatch(stem)
    if match is None:
        return None
    return f"p{int(match.group(2)):02d}"


def _file_stem(task: str) -> str:
    return {
        "ceval-valid": "ceval",
        "gsm8k": "gsm8k",
        "humaneval_instruct": "humaneval",
    }.get(task, task.replace("-", "_"))


def _format_percent(fraction: float) -> str:
    value = round(100.0 * fraction, 2)
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):.0f}%"
    return f"{value:.2f}%"


def _ticks(low, high, count):
    if count < 2:
        return [low]
    step = (high - low) / (count - 1)
    return [low + step * index for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw vector-study figures from result JSON.")
    parser.add_argument("--scores", default=SWEEP_RESULTS)
    parser.add_argument("--tasks", default=BUDGET_RESULTS)
    parser.add_argument("--json", default=JSON_DIR)
    parser.add_argument("--figures", default=FIGURES_DIR)
    args = parser.parse_args()
    for path in write_sweep_plots(args.scores, args.figures):
        print(path)
    for path in write_task_tables(args.tasks, args.json, args.figures):
        print(path)


if __name__ == "__main__":
    main()
