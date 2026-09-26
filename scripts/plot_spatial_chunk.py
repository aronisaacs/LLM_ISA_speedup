#!/usr/bin/env python3
"""Sweep curves and the GSM8K table for the spatial study.

  python scripts/plot_spatial_chunk.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.layer_select.scores import load_sweep_scores, word_perplexity  # noqa: E402
from scripts.plot_vector_study import _format_percent, _table_svg as _budget_table_svg, write_sweep_svg  # noqa: E402

FIGURES = "figures"

_METHODS = (
    ("spatial_pool", "Pooling", "1/8"),
    ("spatial_top1", "Top-1", "2/8"),
    ("spatial_tile", "Per-tile", "2/8"),
    ("spatial_feature", "Per-feature", "2/8"),
)
_STUDY = tuple((method, label) for method, label, _stored in _METHODS)
_PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
_TARGETS = (("k", "Keys"), ("v", "Values"), ("both", "Both"))


def write_wikitext_table(
    results_dir,
    dense_path,
    figures_dir=FIGURES,
) -> Path:
    dense = word_perplexity(json.loads(Path(dense_path).read_text()))
    scores = _load_scores(Path(results_dir))
    rows = [("Baseline", "—", "—", f"{dense:.2f}", "0%", "8/8")]
    for method, method_label, stored in _METHODS:
        for target, target_label in _TARGETS:
            perplexity = scores[(method, target, "pre")]
            rows.append(
                (
                    method_label,
                    target_label,
                    "pre-RoPE",
                    f"{perplexity:.2f}",
                    _degradation(perplexity, dense),
                    stored,
                )
            )
    for method, method_label, stored in _METHODS[:2]:
        perplexity = scores[(method, "k", "post")]
        rows.append(
            (
                method_label,
                "Keys",
                "post-RoPE",
                f"{perplexity:.2f}",
                _degradation(perplexity, dense),
                stored,
            )
        )
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "wikitext.svg"
    path.write_text(_table_svg(rows))
    return path


def _load_scores(results_dir: Path) -> dict[tuple[str, str, str], float]:
    scores = {}
    missing = []
    for method, _label, _stored in _METHODS:
        for target, _target_label in _TARGETS:
            path = results_dir / f"llama31_{method}_{target}.json"
            if not path.is_file():
                missing.append(path.name)
                continue
            scores[(method, target, "pre")] = word_perplexity(json.loads(path.read_text()))
    for method, _label, _stored in _METHODS[:2]:
        path = results_dir / f"llama31_{method}_k_postrope.json"
        if not path.is_file():
            missing.append(path.name)
            continue
        scores[(method, "k", "post")] = word_perplexity(json.loads(path.read_text()))
    if missing:
        raise ValueError(f"missing chunk results: {', '.join(missing)}")
    return scores


def _degradation(perplexity: float, dense: float) -> str:
    if dense == 0:
        return "0%"
    return f"{(perplexity - dense) / dense * 100.0:.2f}%"


def _table_svg(rows) -> str:
    width = 860
    row_h = 36
    header_h = 42
    top = 56
    height = top + header_h + row_h * len(rows) + 16
    columns = (130, 90, 120, 140, 150, 110)
    headers = ("Method", "Target", "RoPE", "Perplexity", "Degradation", "Stored")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="WikiText perplexity for chunk compression">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="18" font-family="system-ui,sans-serif" font-weight="700" '
        'fill="#212529">Llama 3.1 8B Instruct — WikiText chunk compression</text>',
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


def write_wikitext_table_from_index(figures_dir=FIGURES, pretrained="meta-llama/Llama-3.1-8B-Instruct") -> Path:
    """Look up chunk simulations and the dense WikiText baseline in the results index."""
    from engine.eval_runner.index import simulations

    dense = None
    scores = {}
    for record in simulations():
        identity = record.get("identity") or {}
        if "wikitext" not in (identity.get("tasks") or []):
            continue
        if identity.get("pretrained") != pretrained:
            continue
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        value = (record.get("scores") or {}).get("wikitext", {}).get("word_perplexity,none")
        if value is None:
            continue
        if not pipeline:
            dense = float(value)
            continue
        step = pipeline[0]
        method = step.get("method")
        if method not in {name for name, _label, _stored in _METHODS}:
            continue
        target = _target(step)
        rope = "pre" if step.get("pre_rope") else "post"
        if target is None:
            continue
        scores[(method, target, rope)] = float(value)
    if dense is None:
        raise ValueError("no dense WikiText baseline in the index")
    rows = [("Baseline", "—", "—", f"{dense:.2f}", "0%", "8/8")]
    for method, method_label, stored in _METHODS:
        for target, target_label in _TARGETS:
            rows.append(
                (
                    method_label,
                    target_label,
                    "pre-RoPE",
                    f"{scores[(method, target, 'pre')]:.2f}",
                    _degradation(scores[(method, target, "pre")], dense),
                    stored,
                )
            )
    for method, method_label, stored in _METHODS[:2]:
        perplexity = scores[(method, "k", "post")]
        rows.append(
            (
                method_label,
                "Keys",
                "post-RoPE",
                f"{perplexity:.2f}",
                _degradation(perplexity, dense),
                stored,
            )
        )
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "wikitext.svg"
    path.write_text(_table_svg(rows))
    return path


def _target(step: dict) -> str | None:
    k_layers = step.get("k_layers")
    v_layers = step.get("v_layers")
    k_on = k_layers == "all" or k_layers not in (None, [], ())
    v_on = v_layers == "all" or v_layers not in (None, [], ())
    if k_on and v_on:
        return "both"
    if k_on:
        return "k"
    if v_on:
        return "v"
    return None


def write_spatial_study(figures_dir=FIGURES, pretrained=_PRETRAINED) -> list[Path]:
    """One WikiText sweep curve per method, and the GSM8K budget table."""
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for method, label in _STUDY:
        dense_ppl, rows = load_sweep_scores(method=method, pretrained=pretrained)
        n_layers = max(row.slot.layer for row in rows) + 1
        path = destination / f"{method}_sweep.svg"
        write_sweep_svg(
            dense_ppl,
            rows,
            100,
            n_layers,
            path,
            title=f"Per layer resiliency run on Llama 3.1 8B Instruct using WikiText with {label}",
        )
        paths.append(path)
    table = destination / "spatial_gsm8k.svg"
    table.write_text(_budget_table_svg("GSM8K, spatial", _gsm8k_rows(pretrained)))
    paths.append(table)
    return paths


def _gsm8k_rows(pretrained: str) -> list[tuple[str, str, str, str]]:
    from engine.eval_runner.index import simulations

    labels = dict(_STUDY)
    baseline = None
    points = []
    for record in simulations():
        identity = record.get("identity") or {}
        if identity.get("pretrained") != pretrained or "gsm8k" not in (identity.get("tasks") or []):
            continue
        accuracy = (record.get("scores") or {}).get("gsm8k", {}).get("exact_match,flexible-extract")
        if accuracy is None:
            continue
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        if not pipeline:
            if record.get("budget") in (0, 0.0):
                baseline = float(accuracy)
            continue
        method = pipeline[0].get("method")
        if method not in labels or "budget" not in record:
            continue
        points.append(
            (method, float(record["budget"]), float(record.get("compression") or 0.0), float(accuracy))
        )
    if baseline is None:
        raise ValueError("no dense Llama 3.1 GSM8K baseline in the index")
    points.sort(key=lambda item: (list(labels).index(item[0]), item[1]))
    rows = [("Baseline", f"{baseline * 100:.2f}", "0%", "0%")]
    for method, budget, compression, accuracy in points:
        drop = 0.0 if baseline == 0 else (baseline - accuracy) / baseline * 100.0
        rows.append(
            (
                f"{labels[method]} {_format_percent(budget)}",
                f"{accuracy * 100:.2f}",
                f"{drop:.2f}%",
                _format_percent(compression),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the spatial sweep curves and GSM8K table.")
    parser.add_argument("--results", default=None, help="Folder of old uniform-chunk JSONs.")
    parser.add_argument("--dense", default=None)
    parser.add_argument("--figures", default=FIGURES)
    args = parser.parse_args()
    if args.results is None:
        for path in write_spatial_study(args.figures):
            print(path)
        return
    if args.dense is None:
        raise SystemExit("--dense is required with --results")
    print(write_wikitext_table(args.results, args.dense, args.figures))


if __name__ == "__main__":
    main()
