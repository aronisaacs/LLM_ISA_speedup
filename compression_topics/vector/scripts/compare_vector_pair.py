#!/usr/bin/env python3
"""Compare pair-magnitude vector compression with the scalar method on WikiText.

Both sweeps score one key or value layer at a time. The figure is one panel
per prune percent. A missing singleton is left as a gap.

  python compression_topics/vector/scripts/compare_vector_pair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.layer_select.levels import LEVELS  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
_SCALAR = "vector_compress"
_PAIR = "vector_compress_pair"


def write_comparison(out_dir=FIGURES, pretrained=PRETRAINED) -> Path:
    scalar_dense, scalar_rows = load_sweep_scores(method=_SCALAR, pretrained=pretrained)
    pair_dense, pair_rows = load_sweep_scores(method=_PAIR, pretrained=pretrained)
    if abs(scalar_dense - pair_dense) > 1e-6:
        raise ValueError(f"dense baselines differ: scalar {scalar_dense} pair {pair_dense}")
    n_layers = max(row.slot.layer for row in scalar_rows + pair_rows) + 1
    panels = []
    for level in LEVELS:
        panels.append(
            {
                "level": level,
                "scalar_k": _series(scalar_rows, level, "k", n_layers),
                "scalar_v": _series(scalar_rows, level, "v", n_layers),
                "pair_k": _series(pair_rows, level, "k", n_layers),
                "pair_v": _series(pair_rows, level, "v", n_layers),
            }
        )
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "compare_pair.svg"
    path.write_text(_svg(scalar_dense, panels))
    _print_table(scalar_dense, panels)
    return path


def _series(rows, level, target, n_layers):
    values = [None] * n_layers
    for row in rows:
        if row.level == level and row.slot.target == target:
            values[row.slot.layer] = row.ppl
    return values


def _print_table(dense, panels) -> None:
    print(f"dense perplexity  {dense:.3f}")
    print(f"{'prune':>6}  {'slot':<6}  {'layers':>6}  {'scalar':>8}  {'pair':>8}  {'pair-scalar':>12}")
    for panel in panels:
        for target, scalar, pair in (
            ("keys", panel["scalar_k"], panel["pair_k"]),
            ("values", panel["scalar_v"], panel["pair_v"]),
        ):
            both = [(left, right) for left, right in zip(scalar, pair) if left is not None and right is not None]
            count = len(both)
            scalar_mean = sum(left for left, _right in both) / count
            pair_mean = sum(right for _left, right in both) / count
            print(
                f"{panel['level']:>5}%  {target:<6}  {count:>6}  {scalar_mean:8.3f}  "
                f"{pair_mean:8.3f}  {pair_mean - scalar_mean:+12.3f}"
            )


def _svg(dense, panels) -> str:
    width = 960
    panel_h = 250
    height = 48 + panel_h * len(panels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="WikiText perplexity of scalar and pair vector compression">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="64" y="28" font-size="16" font-family="system-ui,sans-serif" font-weight="700" '
        'fill="#212529">Scalar vector compression versus pair magnitude, WikiText</text>',
    ]
    for index, panel in enumerate(panels):
        parts.append(_panel(dense, panel, top=48 + index * panel_h, width=width, height=panel_h))
    parts.append("</svg>")
    return "".join(parts)


def _panel(dense, panel, top, width, height) -> str:
    pad_left, pad_right, pad_top, pad_bottom = 64, 188, 28, 36
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    series = (panel["scalar_k"], panel["scalar_v"], panel["pair_k"], panel["pair_v"])
    samples = [value for values in series for value in values if value is not None]
    samples.append(dense)
    low, high = min(samples), max(samples)
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08
    n_layers = len(panel["scalar_k"])

    def x_at(layer):
        if n_layers == 1:
            return pad_left + plot_w / 2
        return pad_left + plot_w * layer / (n_layers - 1)

    def y_at(value):
        return top + pad_top + plot_h * (1 - (value - low) / (high - low))

    parts = [
        f'<text x="{pad_left}" y="{top + 16}" font-size="13" font-family="system-ui,sans-serif" '
        f'fill="#212529">{panel["level"]}% of each vector</text>'
    ]
    baseline = y_at(dense)
    parts.append(
        f'<line x1="{pad_left}" y1="{baseline:.1f}" x2="{pad_left + plot_w}" y2="{baseline:.1f}" '
        f'stroke="#212529" stroke-width="1.2" stroke-dasharray="6 4"/>'
    )
    styles = (
        (panel["scalar_k"], "#1f77b4", None, "Scalar keys"),
        (panel["scalar_v"], "#ff7f0e", None, "Scalar values"),
        (panel["pair_k"], "#1f77b4", "5 3", "Pair keys"),
        (panel["pair_v"], "#ff7f0e", "5 3", "Pair values"),
    )
    for values, color, dash, _label in styles:
        parts.append(_segments(values, x_at, y_at, color, dash))
    step = 2 if n_layers > 16 else 1
    for layer in range(0, n_layers, step):
        parts.append(
            f'<text x="{x_at(layer):.1f}" y="{top + height - 14}" text-anchor="middle" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#495057">{layer}</text>'
        )
    legend_x = pad_left + plot_w + 16
    legend_y = top + pad_top
    for index, (_values, color, dash, label) in enumerate(styles):
        row_y = legend_y + index * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{row_y}" x2="{legend_x + 22}" y2="{row_y}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>'
            f'<text x="{legend_x + 28}" y="{row_y + 4}" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#212529">{label}</text>'
        )
    parts.append(
        f'<line x1="{legend_x}" y1="{legend_y + 76}" x2="{legend_x + 22}" y2="{legend_y + 76}" '
        f'stroke="#212529" stroke-width="1.2" stroke-dasharray="6 4"/>'
        f'<text x="{legend_x + 28}" y="{legend_y + 80}" font-size="11" '
        f'font-family="system-ui,sans-serif" fill="#212529">Dense</text>'
    )
    return "".join(parts)


def _segments(values, x_at, y_at, color, dash) -> str:
    parts = []
    chunk = []
    for layer, value in enumerate(values):
        if value is None:
            parts.append(_polyline(chunk, color, dash))
            chunk = []
            continue
        chunk.append((x_at(layer), y_at(value)))
    parts.append(_polyline(chunk, color, dash))
    return "".join(parts)


def _polyline(points, color, dash) -> str:
    if not points:
        return ""
    drawn = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}"/>' for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="1.8"{dash_attr} points="{drawn}"/>' + dots


def main() -> None:
    path = write_comparison()
    print(f"wrote  {path}")


if __name__ == "__main__":
    main()
