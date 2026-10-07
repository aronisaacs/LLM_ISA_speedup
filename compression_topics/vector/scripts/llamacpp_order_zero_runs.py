#!/usr/bin/env python3
"""Mean zero runs with keys counted in llama.cpp's feature order.

Hugging Face Llama stores each key head half-split: feature ``i`` rotates with
``i + head_dim // 2``. llama.cpp's converter permutes q/k back to Meta's
interleaved order, where feature ``2i`` rotates with ``2i + 1`` (RoPE type
NORM). Values are never permuted. Sparsification picks the same values in both
orders, so accuracy is unchanged; only where the zeros sit, and so how many
zero runs a vector has, depends on the order.

One Llama 3.1 8B load, then twelve runs on CEval (``limit`` 1 per subject):

- ``scalar``: per-scalar sparsification (vector_compress) at 10-60% on every
  key and value layer, as in vector_zero_runs.py.
- ``pair``: RoPE-paired sparsification (vector_compress_pair) at 10-60% on
  every key layer, values dense, as in vector_pair_zero_runs.py.

Each run records, per vector, zero runs over the 128 values for keys in Hugging
Face order, keys in llama.cpp order and values, plus zero runs over the 64-pair
mask for keys. The log prints each next to the random-placement mean. Scores
are not written to results.json.

  python compression_topics/vector/scripts/llamacpp_order_zero_runs.py
"""

from __future__ import annotations

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
from engine.eval_runner import evaluate, load_model_if_needed, merge, reject_deprecated_kv_keys  # noqa: E402
from engine.kv_compress import install, parse_kv_spec  # noqa: E402
from engine.kv_compress.methods import METHODS  # noqa: E402

PERCENTAGES = (10, 20, 30, 40, 50, 60)
METHOD_NAMES = {"scalar": "vector_compress", "pair": "vector_compress_pair"}
STATS = ("k_hf", "k_llamacpp", "k_pair_mask", "v")
OUT = Path(__file__).resolve().parents[1] / "figures" / "llamacpp_zero_runs.json"


def to_llamacpp_order(keys: torch.Tensor) -> torch.Tensor:
    """Interleave the two halves of each head: HF feature j < d/2 goes to 2j, j + d/2 to 2j + 1.

    Same mapping as llama.cpp's LlamaModel.permute on the k_proj rows.
    """
    half = keys.shape[-1] // 2
    return torch.stack((keys[..., :half], keys[..., half:]), dim=-1).flatten(-2)


def zero_runs(tensor: torch.Tensor) -> int:
    """Maximal exact-zero streaks on the last dimension, summed over every vector."""
    zero = tensor == 0
    previous_zero = F.pad(zero[..., :-1], (1, 0), value=False)
    return int((zero & ~previous_zero).sum().item())


def pair_mask_runs(keys_llamacpp: torch.Tensor) -> int:
    """Zero runs over the pair mask: pair i is zero when values 2i and 2i + 1 both are."""
    pairs = keys_llamacpp.unflatten(-1, (-1, 2))
    return zero_runs((pairs != 0).any(dim=-1).to(torch.int8))


def random_runs(length: int, zeros: int) -> float:
    """Expected zero runs when ``zeros`` of ``length`` positions are zeroed uniformly at random."""
    return zeros * (length - zeros + 1) / length


class Profile:
    def __init__(self) -> None:
        self.runs = dict.fromkeys(STATS, 0)
        self.vectors = dict.fromkeys(STATS, 0)
        self.head_dim = None

    def record(self, tensor: torch.Tensor, target: str) -> None:
        if tensor.numel() == 0:
            return
        vectors = tensor.numel() // tensor.shape[-1]
        with torch.no_grad():
            if target == "v":
                self._add("v", zero_runs(tensor), vectors)
                return
            self.head_dim = tensor.shape[-1]
            ordered = to_llamacpp_order(tensor)
            self._add("k_hf", zero_runs(tensor), vectors)
            self._add("k_llamacpp", zero_runs(ordered), vectors)
            self._add("k_pair_mask", pair_mask_runs(ordered), vectors)

    def _add(self, stat: str, runs: int, vectors: int) -> None:
        self.runs[stat] += runs
        self.vectors[stat] += vectors

    def take(self) -> dict:
        means = {
            stat: (self.runs[stat] / self.vectors[stat]) if self.vectors[stat] else None for stat in STATS
        }
        result = {"mean_zero_runs": means, "vectors": dict(self.vectors), "head_dim": self.head_dim}
        self.__init__()
        return result


def main() -> None:
    base = {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
    }
    profile = Profile()
    originals = {name: METHODS[name] for name in METHOD_NAMES.values()}
    for name, original in originals.items():
        METHODS[name] = _recording(original, profile)
    lm = None
    loaded_model_key = None
    rows = []
    try:
        for mode, configuration in _configurations():
            reject_deprecated_kv_keys(base, configuration)
            pct = configuration["kv"]["pipeline"][0]["prune_pct"]
            kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
            print(f"run  {configuration['name']}", flush=True)
            lm, loaded_model_key, device, model_args = load_model_if_needed(
                lm, loaded_model_key, base, configuration
            )
            uninstall = install(lm, kv_spec)
            try:
                evaluate(lm, base, configuration, kv_spec, device, model_args)
            finally:
                uninstall()
            row = {"mode": mode, "prune_pct": pct, **profile.take()}
            row["random_zero_runs"] = _random_means(mode, pct, row["head_dim"])
            rows.append(row)
            _write(rows)
            _print_row(row)
    finally:
        METHODS.update(originals)
    print("zero runs per vector (measured / random)", flush=True)
    for row in rows:
        _print_row(row)
    print(f"wrote  {OUT}", flush=True)


def _recording(original, profile: Profile):
    def apply(tensor: torch.Tensor, *, target: str, **kwargs) -> torch.Tensor:
        result = original(tensor, target=target, **kwargs)
        profile.record(result, target)
        return result

    return apply


def _configurations() -> list[tuple[str, dict]]:
    extra = {key: value for key, value in CEVAL_VALID_5SHOT.items() if key not in {"name_task", "file"}}
    extra["limit"] = 1
    configurations = []
    for mode in ("scalar", "pair"):
        for pct in PERCENTAGES:
            if mode == "scalar":
                kv = vector_compress(prune_pct=pct)
            else:
                kv = vector_compress_pair(k_layers="all", v_layers=[], prune_pct=pct)
            configuration = {"name": f"llama31_ceval_llamacpp_zero_runs_{mode}_p{pct:02d}", "kv": kv}
            configuration.update(extra)
            configurations.append((mode, configuration))
    return configurations


def _random_means(mode: str, pct: int, head_dim: int | None) -> dict:
    if head_dim is None:
        return {}
    pairs = head_dim // 2
    if mode == "scalar":
        zeroed = (head_dim * pct) // 100
        return {"k_hf": random_runs(head_dim, zeroed), "k_llamacpp": random_runs(head_dim, zeroed),
                "v": random_runs(head_dim, zeroed)}
    zeroed_pairs = (pairs * pct) // 100
    return {"k_pair_mask": random_runs(pairs, zeroed_pairs)}


def _print_row(row: dict) -> None:
    parts = [f"{row['mode']:6s} {row['prune_pct']:2d}%"]
    for stat in STATS:
        measured = row["mean_zero_runs"][stat]
        if measured is None:
            continue
        random_mean = row["random_zero_runs"].get(stat)
        random_text = "" if random_mean is None else f" / {random_mean:.3f}"
        parts.append(f"{stat} {measured:.3f}{random_text}")
    print("  ".join(parts), flush=True)


def _write(rows: list[dict]) -> None:
    payload = {"task": "ceval-valid", "limit": 1, "levels": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
