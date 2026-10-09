#!/usr/bin/env python3
"""Quads with one shared residual mask versus a mask per token, on captured keys and values.

Offline, no model: reads the captures of ``group_rd_offline.py --stage capture``.
A quad stores one mean direction, four norms and r residual entries per token.
  own         each token keeps its own r largest deviations (4 masks); this is
              what group_calibrated and group_rd do today
  own_q       keys only: as own, but deviations weighted by query energy
  shared      the r features with the largest squared deviation summed over
              the 4 tokens are kept for every token (1 mask)
  shared_q    keys only: as shared, weighted by query energy
Positions cost a D-bit mask, or r indices of ceil(log2 D) bits when smaller.

Two comparisons per slot (keys: query-weighted error; values: squared error):
  1. every block a quad with r entries: bits and error of both variants
  2. a per-page menu {dense, quads with r in R}: error at equal stored bits,
     with one price per layer (as group_rd), 64-token pages, 3-bit code

  python compression_topics/spatial/scripts/quad_shared_mask_test.py --captures <dir with plan.json>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.algorithms.storage import metadata_bits
from compression_topics.spatial.scripts import group_rd_offline as offline

UNIFORM_R = (0, 4, 8, 12, 16, 24, 32, 48)
MENU_R = (0, 8, 16, 32)  # dense + 4 quads + spare codes: 3-bit code either way
SAVINGS = (.4, .45, .5, .55, .6, .65, .7)
SLOTS = (("k", "query"), ("v", "squared"))


def position_bits(r, dim):
    return 0 if r == 0 else min(dim, r * math.ceil(math.log2(dim)))


def quad_bits(r, dim, shared):
    """Mean + r entries per token + 4 norms, all 16-bit, plus residual positions."""
    return 16 * (dim + 4 * r + 4) + (1 if shared else 4) * position_bits(r, dim)


def quad_errors(body, rope, kind, weights, residuals):
    """Per block and r: error of per-token and shared-mask quads. body is [1, H, T, D]."""
    _, heads, length, dim = body.shape
    blocks = length // 4
    original = body.float()
    norms = original.norm(dim=-1, keepdim=True).reshape(1, heads, blocks, 4, 1)
    unit = torch.nn.functional.normalize(group_rd._aligned(original, 4, rope), dim=-1)
    unit = unit.reshape(1, heads, blocks, 4, dim)
    mean = unit.mean(-2, keepdim=True)
    spread = unit - mean
    w = None
    if kind == "query":
        w = group_rd.plane_tied(weights.float()).to(body.device).reshape(1, heads, 1, 1, dim)
    top = max(residuals)
    squared = spread.square()
    indices = {"own": squared.topk(top, dim=-1).indices,
               "shared": squared.sum(-2).topk(top, dim=-1).indices}  # shared: [1, H, B, top]
    if w is not None:
        indices["own_q"] = (squared * w).topk(top, dim=-1).indices
        indices["shared_q"] = (squared * w).sum(-2).topk(top, dim=-1).indices
    out = {variant: [] for variant in indices}
    for r in residuals:
        for variant, chosen in indices.items():
            residual = torch.zeros_like(spread)
            if r:
                index = chosen[..., :r]
                if variant.startswith("shared"):
                    index = index.unsqueeze(-2).expand(*spread.shape[:-1], r)
                residual.scatter_(-1, index, spread.gather(-1, index))
            error, _ = group_rd._distortion(mean + residual, unit, norms, kind, w)
            out[variant].append(error.sum(-1).double().cpu())  # [1, H, B]
    return {variant: torch.stack(rows, -1) for variant, rows in out.items()}  # [1, H, B, R]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=offline.OUT,
                        help="folder with plan.json, rope.json, query_weights.pt and keys/")
    parser.add_argument("--chunks", type=int, default=None, help="use only the first N captured chunks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="JSON output (default <captures>/quad_shared_mask.json)")
    args = parser.parse_args()
    manifest = json.loads((args.captures / "plan.json").read_text())
    rope = offline.load_rope(manifest["rope"])
    weights = torch.load(manifest["query_weights"], weights_only=True)["weights"]
    files = sorted(Path(manifest["keys_dir"]).glob("chunk_*.pt"))[:args.chunks]
    if not files:
        raise SystemExit("no captured chunks")
    residuals = tuple(sorted(set(UNIFORM_R) | set(MENU_R)))
    errors = {slot: {} for slot in SLOTS}  # slot -> layer -> variant -> [chunk tensors]
    for path in files:
        captured = torch.load(path, weights_only=True)
        layers, heads, length, dim = captured["keys"].shape
        full = length // 64 * 64
        for target, kind in SLOTS:
            source = captured["keys" if target == "k" else "values"]
            for layer in range(layers):
                body = source[layer, :, :full].unsqueeze(0).to(args.device)
                result = quad_errors(body, rope if target == "k" else None, kind,
                                     weights[layer] if kind == "query" else None, residuals)
                for variant, table in result.items():
                    errors[target, kind].setdefault(layer, {}).setdefault(variant, []).append(table)
        print(f"[quad mask] {path.name}", flush=True)
    dense_block = 4 * dim * 16
    report = {"model_args": manifest["model_args"], "chunks": len(files), "head_dim": dim, "layers": layers,
              "uniform": {}, "menu": {}}
    lines = [f"# Shared vs per-token quad masks ({manifest['model_args'].split(',')[0].split('=')[1]}, "
             f"{len(files)} chunks, head dim {dim})", ""]
    for target, kind in SLOTS:
        name = f"{'keys' if target == 'k' else 'values'} ({kind} error)"
        # 1. Uniform quads: every block in Qr.
        variants = list(next(iter(errors[target, kind].values())))
        others = [v for v in variants if v != "own"]
        rows = []
        for r in UNIFORM_R:
            i = residuals.index(r)
            ratios = {v: statistics.median(float(sum(t[..., i].sum() for t in by[v]) / sum(t[..., i].sum() for t in by["own"]))
                                           for by in errors[target, kind].values()) for v in others}
            rows.append({"r": r, "bits_own": quad_bits(r, dim, False), "bits_shared": quad_bits(r, dim, True),
                         "saving_own": 1 - quad_bits(r, dim, False) / dense_block,
                         "saving_shared": 1 - quad_bits(r, dim, True) / dense_block, "error_ratio": ratios})
        report["uniform"][f"{target}/{kind}"] = rows
        lines += [f"## {name}: every block a quad (error relative to today's per-token masks)", "",
                  "| r | bits per-token | bits shared | saving per-token | saving shared | " + " | ".join(others) + " |",
                  "|---|---|---|---|---|" + "---|" * len(others)]
        lines += [f"| {x['r']} | {x['bits_own']} | {x['bits_shared']} | {x['saving_own']:.1%} | {x['saving_shared']:.1%} | "
                  + " | ".join(f"{x['error_ratio'][v]:.2f}×" for v in others) + " |" for x in rows]
        lines.append("")
        # 2. Per-page menus at equal bits.
        menu_idx = [residuals.index(r) for r in MENU_R]
        gains = {v: {s: [] for s in SAVINGS} for v in others}
        for layer, by in errors[target, kind].items():
            curves = {}
            for variant in variants:
                bits = torch.tensor([dense_block] + [quad_bits(r, dim, variant == "shared") for r in MENU_R],
                                    dtype=torch.float64)
                tables = [torch.cat((torch.zeros_like(t[..., :1]), t[..., menu_idx]), -1) for t in by[variant]]
                blocks = tables[0].shape[-2]
                overhead = metadata_bits(heads * blocks // 16, 3)
                lams = offline._lambdas(tables, bits, 1)
                d, b = offline.curve(tables, bits, lams, 64, overhead)
                dense = len(tables) * heads * blocks * dense_block
                curves[variant] = ((1 - b / dense).tolist(), d.tolist())
            own = offline.envelope(*curves["own"], SAVINGS)
            for v in others:
                for s, a, c in zip(SAVINGS, own, offline.envelope(*curves[v], SAVINGS)):
                    if a and c is not None:
                        gains[v][s].append(1 - c / a)
        report["menu"][f"{target}/{kind}"] = {v: {str(s): statistics.median(g) if g else None for s, g in by_s.items()}
                                              for v, by_s in gains.items()}
        lines += [f"## {name}: per-page menu dense + Q{{{', '.join(map(str, MENU_R))}}}, equal stored bits", "",
                  "Error reduction vs today's per-token masks, median over layers (layers improved).", "",
                  "| saving | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for s in SAVINGS:
            cells = []
            for v in others:
                g = gains[v][s]
                cells.append(f"{statistics.median(g):+.1%} ({sum(x > 0 for x in g)}/{len(g)})" if g else "–")
            lines.append(f"| {s:.0%} | " + " | ".join(cells) + " |")
        lines.append("")
    out = args.out or args.captures / "quad_shared_mask.json"
    out.write_text(json.dumps(report, indent=1))
    (out.with_suffix(".md")).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
