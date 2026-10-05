#!/usr/bin/env python3
"""Net compression of each GSM8K budget run once RLE metadata is paid for.

The budget runs mix levels per slot (25/50/75% of a K or V vector in a given
layer), so the RLE cost differs slot by slot. This script captures post-RoPE K
and V for Llama 3.1 8B on a laptop (llamacpp_zero_groups_local.capture), counts
zero-groups for every (layer, K/V, level) in llama.cpp's key layout, then walks
each per-scalar and RoPE-paired GSM8K row in results.json:

  net = sum over compressed slots of (16 x values zeroed - 16 x zero-groups)
        / (64 slots x 128 values x 16 bits)

Per-scalar slots and RoPE-paired V slots count groups over the 128 values;
RoPE-paired K slots count groups over the 64 pairs (one RLE list for both
values of a pair). Uncompressed slots cost nothing.

  python compression_topics/vector/scripts/rle_net_compression.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.vector.algorithms.vector_compress import _apply as scalar_sparsify  # noqa: E402
from compression_topics.vector.algorithms.vector_compress_pair import _prune_rope_pairs as pair_sparsify  # noqa: E402
from compression_topics.vector.scripts.llamacpp_zero_groups import to_llamacpp_order, zero_group_starts  # noqa: E402

LEVELS = (25, 50, 75)
FIGURES = Path(__file__).resolve().parents[1] / "figures"
SLOT_OUT = FIGURES / "rle_slot_groups.json"
NET_OUT = FIGURES / "rle_net_compression.json"
METHODS = ("vector_compress", "vector_compress_pair")
HEAD_DIM = 128
VALUE_BITS = 16
RLE_BITS = 16


def groups_per_vector(tensor: torch.Tensor) -> float:
    return float(zero_group_starts(tensor).sum()) / (tensor.numel() // tensor.shape[-1])


def slot_groups(keys: dict, values: dict) -> dict:
    """Zero-groups per vector for every layer, target and level, llama.cpp key order."""
    table = {"scalar_k": {}, "scalar_v": {}, "pair_k": {}}
    for layer in sorted(keys):
        k, v = keys[layer].float(), values[layer].float()
        for name in table:
            table[name][str(layer)] = {}
        for level in LEVELS:
            ks = to_llamacpp_order(scalar_sparsify(k, threshold=0.0, prune_pct=level))
            table["scalar_k"][str(layer)][str(level)] = groups_per_vector(ks)
            table["scalar_v"][str(layer)][str(level)] = groups_per_vector(
                scalar_sparsify(v, threshold=0.0, prune_pct=level)
            )
            kp = to_llamacpp_order(pair_sparsify(k, level))
            pair_mask = (kp.reshape(*kp.shape[:-1], -1, 2) != 0).any(-1).float()
            table["pair_k"][str(layer)][str(level)] = groups_per_vector(pair_mask)
    return table


def slot_net_bits(method: str, target: str, layer: int, level: int, table: dict) -> tuple[float, float]:
    """(bits removed, RLE bits) for one compressed slot, per vector."""
    if method == "vector_compress_pair" and target == "k":
        zeroed_values = 2 * ((HEAD_DIM // 2) * level // 100)
        groups = table["pair_k"][str(layer)][str(level)]
    else:
        zeroed_values = HEAD_DIM * level // 100
        groups = table[f"scalar_{target}"][str(layer)][str(level)]
    return VALUE_BITS * zeroed_values, RLE_BITS * groups


def run_compression(pipeline: list[dict], method: str, table: dict, n_layers: int = 32) -> dict:
    removed = rle = 0.0
    for step in pipeline:
        level = int(step["prune_pct"])
        for target in ("k", "v"):
            for layer in step.get(f"{target}_layers") or []:
                r, m = slot_net_bits(method, target, int(layer), level, table)
                removed += r
                rle += m
    total = 2 * n_layers * HEAD_DIM * VALUE_BITS
    return {"nominal": removed / total, "rle_overhead": rle / total, "net": (removed - rle) / total}


def gsm8k_rows(results_path: Path) -> list[dict]:
    payload = json.loads(results_path.read_text())
    rows = []
    for row in payload["simulations"]:
        identity = row["identity"]
        if identity.get("tasks") != ["gsm8k"] or identity.get("limit"):
            continue
        if identity.get("pretrained") != "meta-llama/Llama-3.1-8B-Instruct":
            continue
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        methods = {step["method"] for step in pipeline}
        if len(methods) != 1 or not methods <= set(METHODS):
            continue
        rows.append(
            {
                "method": methods.pop(),
                "budget": row.get("budget"),
                "accuracy": row["scores"]["gsm8k"]["exact_match,flexible-extract"],
                "pipeline": pipeline,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=("gsm8k", "ceval"), default="ceval")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--reuse", action="store_true", help="reuse figures/rle_slot_groups.json")
    args = parser.parse_args()

    if args.reuse and SLOT_OUT.is_file():
        table = json.loads(SLOT_OUT.read_text())["groups_per_vector"]
    else:
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoTokenizer

        from compression_topics.vector.scripts import llamacpp_zero_groups_local as local

        model_dir = Path(try_to_load_from_cache(local.MODEL, "model.safetensors.index.json")).parent
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        input_ids = local.load_text(args.text, tokenizer, args.tokens, 512)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        recorder = local.capture(model_dir, input_ids, device)
        table = slot_groups(recorder.keys, recorder.values)
        FIGURES.mkdir(parents=True, exist_ok=True)
        SLOT_OUT.write_text(
            json.dumps(
                {"text": args.text, "tokens": int(input_ids.numel()), "layout": "llama.cpp", "groups_per_vector": table},
                indent=2,
            )
            + "\n"
        )

    out = []
    for row in gsm8k_rows(ROOT / "results.json"):
        net = run_compression(row["pipeline"], row["method"], table)
        out.append({k: row[k] for k in ("method", "budget", "accuracy")} | net)
    out.sort(key=lambda r: (r["method"], r["budget"] or 0))
    NET_OUT.write_text(json.dumps({"rows": out}, indent=2) + "\n")
    for r in out:
        print(
            f"{r['method']:22s} budget {r['budget']:.2f}  nominal {r['nominal']:.3f}  "
            f"RLE {r['rle_overhead']:.3f}  net {r['net']:.3f}  acc {r['accuracy']:.3f}",
            flush=True,
        )
    print(f"wrote  {NET_OUT}", flush=True)


if __name__ == "__main__":
    main()
