#!/usr/bin/env python3
"""Zero-group statistics in llama.cpp's key layout, for per-scalar and RoPE-paired.

transformers stores each Llama key head as two halves, so RoPE pairs are
(i, i + d/2). llama.cpp's GGUF converter permutes the q/k weights back to
Meta's order, where RoPE pairs are adjacent, (2i, 2i + 1). Values are not
permuted. Sparsification picks the same values in either layout; only the
positions of the zeros, and so the zero-groups RLE has to store, differ.

This script runs the model in transformers as usual, then, for every K chunk
compression produces, reorders it into llama.cpp's layout before counting
zero-groups. V chunks are counted as they are.

  per-scalar  vector_compress on every K and V layer; groups over 128 values
  pair        vector_compress_pair on every K layer (V dense); groups over the
              64 pairs and over the 128 values (the two agree, since pairs are
              zeroed together and are adjacent in llama.cpp's layout)

One Llama 3.1 8B load, CEval 5-shot, limit 1 (one question per subject), the
same setup as vector_zero_runs.py and vector_pair_zero_runs.py. Scores are not
written to results.json.

  python compression_topics/vector/scripts/llamacpp_zero_groups.py
  python compression_topics/vector/scripts/llamacpp_zero_groups.py --method scalar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress, vector_compress_pair  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT  # noqa: E402

PERCENTAGES = (10, 20, 30, 40, 50, 60)
METHODS = ("scalar", "pair")
OUT = Path(__file__).resolve().parents[1] / "figures" / "llamacpp_zero_groups.json"
RLE_BITS_PER_GROUP = 16  # 1-byte start, 1-byte length
VALUE_BITS = 16  # bf16


def to_llamacpp_order(keys: torch.Tensor) -> torch.Tensor:
    """transformers key layout -> llama.cpp: position 2j = j, position 2j + 1 = j + d/2."""
    half = keys.shape[-1] // 2
    return torch.stack((keys[..., :half], keys[..., half:]), dim=-1).flatten(-2)


def zero_group_starts(tensor: torch.Tensor) -> torch.Tensor:
    """Bool mask of positions that start a run of exact zeros along the last dimension."""
    zero = tensor == 0
    previous_zero = F.pad(zero[..., :-1], (1, 0), value=False)
    return zero & ~previous_zero


def random_groups(length: int, zeroed: int) -> float:
    """Expected zero-groups when ``zeroed`` of ``length`` positions are chosen uniformly."""
    return zeroed * (length - zeroed + 1) / length


class Tally:
    """Running totals for one target, kept on the tensor's device until read."""

    def __init__(self) -> None:
        self.groups = None
        self.zeros = None
        self.vectors = 0

    def add(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        with torch.no_grad():
            groups = zero_group_starts(tensor).sum()
            zeros = (tensor == 0).sum()
        self.groups = groups if self.groups is None else self.groups + groups.to(self.groups.device)
        self.zeros = zeros if self.zeros is None else self.zeros + zeros.to(self.zeros.device)
        self.vectors += tensor.numel() // tensor.shape[-1]

    def summary(self) -> dict | None:
        if not self.vectors:
            return None
        groups = int(self.groups.item())
        zeros = int(self.zeros.item())
        return {
            "vectors": self.vectors,
            "groups_per_vector": groups / self.vectors,
            "zeros_per_vector": zeros / self.vectors,
            "avg_group_length": zeros / groups if groups else None,
        }


class Profiler:
    """Wraps the cache's compress_kv and tallies its outputs in llama.cpp order."""

    def __init__(self, count_values: bool) -> None:
        self.count_values = count_values
        self.k = Tally()
        self.k_pairs = Tally()
        self.v = Tally()
        self._original = None

    def install(self) -> None:
        import engine.kv_compress.cache as cache_module

        self._original = cache_module.compress_kv
        original = self._original
        profiler = self

        def compress_kv(key_states, value_states, layer_idx, spec, *args, **kwargs):
            keys, values = original(key_states, value_states, layer_idx, spec, *args, **kwargs)
            profiler.record(keys, values)
            return keys, values

        cache_module.compress_kv = compress_kv

    def uninstall(self) -> None:
        import engine.kv_compress.cache as cache_module

        if self._original is not None:
            cache_module.compress_kv = self._original
            self._original = None

    def record(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        cpp_keys = to_llamacpp_order(keys)
        self.k.add(cpp_keys)
        # Pair mask: a pair is zero when both of its values are. In llama.cpp order
        # the pair is (2i, 2i + 1), so view the key as [..., 64, 2].
        pairs = cpp_keys.reshape(*cpp_keys.shape[:-1], -1, 2)
        self.k_pairs.add((pairs != 0).any(dim=-1).to(cpp_keys.dtype))
        if self.count_values:
            self.v.add(values)


def configurations(method: str) -> list[dict]:
    extra = {key: value for key, value in CEVAL_VALID_5SHOT.items() if key not in {"name_task", "file"}}
    extra["limit"] = 1
    out = []
    for pct in PERCENTAGES:
        if method == "scalar":
            kv = vector_compress(prune_pct=pct)
        else:
            kv = vector_compress_pair(k_layers="all", v_layers=[], prune_pct=pct)
        configuration = {"name": f"llama31_ceval_llamacpp_groups_{method}_p{pct:02d}", "kv": kv}
        configuration.update(extra)
        out.append(configuration)
    return out


def level_row(method: str, pct: int, profiler: Profiler, head_dim: int = 128) -> dict:
    """Measured groups next to random placement, plus the RLE saving they imply."""
    k = profiler.k.summary()
    row = {"sparsity_pct": pct, "k": k}
    if method == "scalar":
        zeroed = head_dim * pct // 100
        v = profiler.v.summary()
        row["v"] = v
        row["random_groups_per_vector"] = random_groups(head_dim, zeroed)
        row["zeroed_per_vector"] = zeroed
        for target, stats in (("k", k), ("v", v)):
            if stats:
                stats["rle_net_saving"] = rle_net_saving(zeroed, stats["groups_per_vector"], head_dim)
    else:
        pairs = head_dim // 2
        zeroed_pairs = pairs * pct // 100
        k_pairs = profiler.k_pairs.summary()
        if k_pairs:
            # The pair tally counts zero *pairs*; groups over pairs equal groups over values.
            k_pairs["avg_group_length_pairs"] = (
                zeroed_pairs / k_pairs["groups_per_vector"] if k_pairs["groups_per_vector"] else None
            )
            k_pairs["rle_net_saving"] = rle_net_saving(2 * zeroed_pairs, k_pairs["groups_per_vector"], head_dim)
        row["k_pairs"] = k_pairs
        row["zeroed_pairs_per_vector"] = zeroed_pairs
        row["random_groups_per_vector"] = random_groups(pairs, zeroed_pairs)
    return row


def rle_net_saving(values_zeroed: float, groups: float, head_dim: int = 128) -> float:
    """(bits freed - RLE bits) / uncompressed bits, per vector."""
    return (VALUE_BITS * values_zeroed - RLE_BITS_PER_GROUP * groups) / (VALUE_BITS * head_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=(*METHODS, "both"), default="both")
    args = parser.parse_args()
    methods = METHODS if args.method == "both" else (args.method,)

    from engine.eval_runner import evaluate, load_model_if_needed, merge, reject_deprecated_kv_keys
    from engine.kv_compress import install, parse_kv_spec

    base = {"model": "hf", "batch_size": BATCH_SIZE, "model_args": LLAMA31_8B}
    payload = _read_existing()
    lm = None
    loaded_model_key = None
    for method in methods:
        rows = []
        for configuration in configurations(method):
            reject_deprecated_kv_keys(base, configuration)
            pct = configuration["kv"]["pipeline"][0]["prune_pct"]
            kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
            print(f"run  {configuration['name']}", flush=True)
            lm, loaded_model_key, device, model_args = load_model_if_needed(lm, loaded_model_key, base, configuration)
            profiler = Profiler(count_values=(method == "scalar"))
            uninstall = install(lm, kv_spec)
            profiler.install()
            try:
                evaluate(lm, base, configuration, kv_spec, device, model_args)
            finally:
                profiler.uninstall()
                uninstall()
            row = level_row(method, pct, profiler)
            rows.append(row)
            payload[method] = rows
            _write(payload)
            _print_row(method, row)
    print(f"wrote  {OUT}", flush=True)


def _print_row(method: str, row: dict) -> None:
    pct = row["sparsity_pct"]
    rand = row["random_groups_per_vector"]
    if method == "scalar":
        k, v = row["k"], row["v"]
        print(
            f"scalar {pct}%  K groups {k['groups_per_vector']:.3f}  V groups {v['groups_per_vector']:.3f}  "
            f"random {rand:.3f}  K len {k['avg_group_length']:.3f}  V len {v['avg_group_length']:.3f}",
            flush=True,
        )
    else:
        kp = row["k_pairs"]
        print(
            f"pair   {pct}%  K pair-groups {kp['groups_per_vector']:.3f}  random {rand:.3f}  "
            f"len {kp['avg_group_length_pairs']:.3f} pairs  K value-groups {row['k']['groups_per_vector']:.3f}",
            flush=True,
        )


def _read_existing() -> dict:
    if OUT.is_file():
        try:
            return json.loads(OUT.read_text())
        except json.JSONDecodeError:
            pass
    return {"task": "ceval-valid", "limit": 1, "layout": "llama.cpp (adjacent RoPE pairs)"}


def _write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
