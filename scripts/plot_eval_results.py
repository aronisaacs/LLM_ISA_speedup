#!/usr/bin/env python3
"""Turn multi_run.py result JSONs into a dense-vs-compression accuracy chart.

Reads lm-eval dumps listed in results/index.json, labels each configuration from its kv.pipeline
(dense vs sparsify n:m, etc.), and writes an SVG bar chart plus a text table.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PRIMARY_METRICS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
)


def load_runs(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        results = payload.get("results") or {}
        config = payload.get("config") or {}
        model_args = config.get("model_args") or {}
        kv = {}
        if isinstance(model_args, dict):
            kv = model_args.get("kv") or {}
        n_samples = payload.get("n-samples") or {}
        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            metric_name, value, stderr = _pick_metric(metrics)
            if metric_name is None:
                continue
            sample_info = n_samples.get(task) or {}
            rows.append(
                {
                    "file": str(path),
                    "task": task,
                    "condition": _condition_label(kv),
                    "metric": metric_name,
                    "value": float(value),
                    "stderr": None if stderr is None else float(stderr),
                    "n": sample_info.get("effective") or sample_info.get("original"),
                    "model": _model_name(model_args, config),
                    "kv": kv,
                }
            )
    return rows


def _model_name(model_args, config) -> str:
    if isinstance(model_args, dict) and model_args.get("pretrained"):
        return str(model_args["pretrained"])
    if isinstance(config.get("model"), str):
        return config["model"]
    return "unknown"


def _condition_label(kv: dict) -> str:
    pipeline = kv.get("pipeline") if isinstance(kv, dict) else None
    if not pipeline:
        return "dense"
    parts = []
    for step in pipeline:
        method = step.get("method", "?")
        if method == "sparsify_nm":
            parts.append(f"sparsify {step.get('n', '?')}:{step.get('m', '?')}")
        else:
            parts.append(str(method))
    return " + ".join(parts) if parts else "dense"


def _pick_metric(metrics: dict) -> tuple[str | None, float | None, float | None]:
    for name in PRIMARY_METRICS:
        if name in metrics and metrics[name] is not None:
            stderr = metrics.get(f"{name.split(',')[0]}_stderr,{name.split(',', 1)[1]}")
            if stderr is None:
                stderr = metrics.get(name.replace(",", "_stderr,", 1))
            return name, metrics[name], stderr
    for key, value in metrics.items():
        if key in {"alias"} or "stderr" in key or not isinstance(value, (int, float)):
            continue
        stderr_key = key.replace(",", "_stderr,", 1) if "," in key else None
        stderr = metrics.get(stderr_key) if stderr_key else None
        return key, value, stderr
    return None, None, None


def write_svg(rows: list[dict], output: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)

    tasks = sorted(grouped)
    conditions = []
    for row in rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])

    width = max(720, 220 * len(tasks) + 80)
    height = 420
    pad_left, pad_right, pad_top, pad_bottom = 56, 24, 48, 64
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    colors = ["#3b5bdb", "#c92a2a", "#2b8a3e", "#e67700"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Accuracy by task and KV compression">'
        f'<rect width="100%" height="100%" fill="#f8f9fa"/>'
        f'<text x="{pad_left}" y="28" font-size="16" font-family="system-ui,sans-serif" fill="#212529">'
        f"Accuracy vs KV compression</text>"
        f'<text x="{pad_left}" y="44" font-size="11" font-family="system-ui,sans-serif" fill="#495057">'
        f"Error bars are lm-eval stderr. More compression is expected to lower accuracy, not raise it.</text>"
    ]

    # y axis 0-1
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_top + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left + plot_w}" y2="{y:.1f}" '
            f'stroke="#dee2e6" stroke-width="1"/>'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#495057">{frac:.2f}</text>'
        )

    group_w = plot_w / max(len(tasks), 1)
    bar_w = min(36, (group_w * 0.7) / max(len(conditions), 1))
    for task_i, task in enumerate(tasks):
        group_x = pad_left + group_w * task_i + group_w * 0.15
        by_cond = {row["condition"]: row for row in grouped[task]}
        for cond_i, cond in enumerate(conditions):
            row = by_cond.get(cond)
            if row is None:
                continue
            x = group_x + cond_i * (bar_w + 6)
            h = plot_h * max(0.0, min(1.0, row["value"]))
            y = pad_top + plot_h - h
            color = colors[cond_i % len(colors)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}">'
                f'<title>{task} {cond}: {row["value"]:.3f}</title></rect>'
            )
            if row["stderr"] is not None and not math.isnan(row["stderr"]):
                err = plot_h * row["stderr"]
                mid = x + bar_w / 2
                y1 = max(pad_top, y - err)
                y2 = min(pad_top + plot_h, y + h + err)
                parts.append(
                    f'<line x1="{mid:.1f}" y1="{y1:.1f}" x2="{mid:.1f}" y2="{y2:.1f}" '
                    f'stroke="#212529" stroke-width="1"/>'
                )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="10" '
                f'font-family="system-ui,sans-serif" fill="#212529">{row["value"]:.2f}</text>'
            )
        parts.append(
            f'<text x="{pad_left + group_w * task_i + group_w / 2:.1f}" y="{height - 28}" '
            f'text-anchor="middle" font-size="12" font-family="system-ui,sans-serif" fill="#212529">'
            f"{task}</text>"
        )

    legend_x = pad_left
    for cond_i, cond in enumerate(conditions):
        color = colors[cond_i % len(colors)]
        parts.append(
            f'<rect x="{legend_x}" y="{height - 18}" width="10" height="10" fill="{color}"/>'
            f'<text x="{legend_x + 14}" y="{height - 9}" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#212529">{cond}</text>'
        )
        legend_x += 18 + 7 * len(cond)

    parts.append("</svg>")
    output.write_text("".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        default=None,
        help="Result JSON files or globs. Default: every JSON in the results index.",
    )
    parser.add_argument("-o", "--output", default="results/figures/accuracy_vs_compression.svg")
    args = parser.parse_args()

    paths: list[Path] = []
    if not args.inputs:
        from engine.eval_runner.index import results_root, simulations

        root = results_root()
        for record in simulations():
            stored = record.get("path") or ""
            path = Path(stored) if Path(stored).is_absolute() else root / stored
            if path.is_file():
                paths.append(path)
    for item in args.inputs or []:
        match = list(Path().glob(item)) if any(ch in item for ch in "*?[]") else [Path(item)]
        paths.extend(path for path in match if path.is_file() and path.suffix == ".json")
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit(f"No JSON results found for {args.inputs}")

    rows = load_runs(paths)
    if not rows:
        raise SystemExit("JSON files had no plottable metrics")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_svg(rows, output)

    print(f"Wrote {output} from {len(paths)} files")
    print(f"{'task':<16} {'condition':<22} {'metric':<28} {'acc':>6} {'n':>5}")
    for row in sorted(rows, key=lambda r: (r["task"], r["condition"])):
        n = "" if row["n"] is None else str(row["n"])
        print(f"{row['task']:<16} {row['condition']:<22} {row['metric']:<28} {row['value']:>6.3f} {n:>5}")


if __name__ == "__main__":
    main()
