#!/usr/bin/env python3

import argparse
import heapq
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--tile", type=int, default=None)
    parser.add_argument("--pcts", default=None)
    parser.add_argument(
        "--scope",
        choices=("per-target", "global"),
        default="per-target",
        help=(
            "Allocate the requested average independently to K and V "
            "(per-target, the default), or across one shared K/V pool (global)"
        ),
    )
    parser.add_argument("--higher-is-better", action="store_true")
    return parser.parse_args()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_pcts(raw, rows):
    if raw:
        return sorted(int(x) for x in raw.split(",") if x.strip())
    return sorted({int(row["pct"]) for row in rows if row.get("target") in ("k", "v")})


def build_chunks(metrics, pcts, baseline, higher_is_better):
    anchors = [0] + pcts
    ratios = {target: {layer: 0.0 for layer in metrics[target]} for target in ("k", "v")}
    chunks = {"k": {}, "v": {}}

    for target in ("k", "v"):
        for layer, pct_values in metrics[target].items():
            prev_delta = 0.0
            chunks[target][layer] = []
            for idx, pct in enumerate(pcts):
                if pct not in pct_values:
                    raise ValueError(
                        f"Missing {pct}% measurement for {target.upper()} layer {layer}"
                    )
                value = pct_values[pct]
                delta = (
                    max(0.0, baseline - value)
                    if higher_is_better
                    else max(0.0, value - baseline)
                )
                delta = max(delta, prev_delta)
                chunk = (anchors[idx + 1] - anchors[idx]) / 100.0
                cost = (delta - prev_delta) / chunk
                chunks[target][layer].append((cost, chunk))
                prev_delta = delta

    return ratios, chunks


def allocate_pool(targets, ratios, chunks, budget):
    heap = []
    for target in targets:
        for layer, layer_chunks in chunks[target].items():
            if layer_chunks:
                heapq.heappush(heap, (layer_chunks[0][0], 0, -layer, target, layer))

    remaining = budget
    while remaining > 1e-12 and heap:
        _cost, idx, _neg_layer, target, layer = heapq.heappop(heap)
        _cost, chunk = chunks[target][layer][idx]
        take = min(chunk, remaining)
        ratios[target][layer] += take
        remaining -= take
        if take >= chunk - 1e-12 and idx + 1 < len(chunks[target][layer]):
            next_cost = chunks[target][layer][idx + 1][0]
            heapq.heappush(heap, (next_cost, idx + 1, -layer, target, layer))

    if remaining > 1e-12:
        available = budget - remaining
        raise ValueError(
            f"Requested allocation budget {budget:.6g} exceeds the available "
            f"budget {available:.6g} for scope {','.join(targets)}"
        )


def main():
    args = parse_args()
    if args.target < 0.0:
        raise ValueError("--target must be non-negative")
    input_path = Path(args.input)
    rows = load_rows(input_path)
    pcts = parse_pcts(args.pcts, rows)
    baseline = next(
        float(row["value"])
        for row in rows
        if row.get("target") == "baseline" and (args.task is None or row.get("task") == args.task)
    )

    metrics = {"k": {}, "v": {}}
    for row in rows:
        target = row.get("target")
        if target not in metrics:
            continue
        if args.task is not None and row.get("task") != args.task:
            continue
        if args.tile is not None and int(row["tile"]) != args.tile:
            continue
        metrics[target].setdefault(int(row["layer"]), {})[int(row["pct"])] = float(row["value"])

    ratios, chunks = build_chunks(metrics, pcts, baseline, args.higher_is_better)
    if args.scope == "per-target":
        for target in ("k", "v"):
            allocate_pool(
                (target,),
                ratios,
                chunks,
                args.target * len(ratios[target]),
            )
    else:
        total_layers = sum(len(ratios[target]) for target in ("k", "v"))
        allocate_pool(("k", "v"), ratios, chunks, args.target * total_layers)

    output = {"tile_sparsity_prune_pct_by_target_layer": {}}
    for target in ("k", "v"):
        output["tile_sparsity_prune_pct_by_target_layer"][target] = {
            str(layer): int(round(ratio * 100))
            for layer, ratio in sorted(ratios[target].items())
            if ratio > 1e-12
        }

    output_path = Path(args.output) if args.output else input_path.with_name("greedykv_allocation.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
