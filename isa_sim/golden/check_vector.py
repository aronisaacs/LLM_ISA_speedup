#!/usr/bin/env python3
"""Check llama.cpp's vck50/vcv50 cache types against the PyTorch vector methods.

Keys go through ``vector_compress_pair`` at 50% (half the RoPE pairs dropped)
and values through ``vector_compress`` at 50% (half the scalars dropped), the
same functions the accuracy studies call. llama.cpp keys are in GGUF layout,
where RoPE pairs are adjacent (2i, 2i + 1); the PyTorch code expects the
transformers layout, pairs (i, i + 64). Keys are converted to transformers
layout, compressed there, and converted back, so the check also covers the
layout difference.

Inputs are rounded to fp16 first, so the kept values must match exactly. The
vec_dot check compares llama.cpp's q.k over the packed key with q.k over the
PyTorch reconstruction.

  conda activate llm_isa_local
  python isa_sim/golden/check_vector.py --bin isa_sim/golden/build/vc_roundtrip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.vector.algorithms.vector_compress import _apply  # noqa: E402
from compression_topics.vector.algorithms.vector_compress_pair import _prune_rope_pairs  # noqa: E402

DIM = 128
PRUNE_PCT = 50


def to_llamacpp_order(x: torch.Tensor) -> torch.Tensor:
    """transformers layout -> llama.cpp: position 2j = j, position 2j + 1 = j + d/2."""
    half = x.shape[-1] // 2
    return torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)


def to_hf_order(x: torch.Tensor) -> torch.Tensor:
    """llama.cpp layout -> transformers layout."""
    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)


def sample(n: int, seed: int) -> torch.Tensor:
    """Gaussian vectors with per-channel scales, so some channels are outliers like real K/V."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(DIM, generator=g))
    x = torch.randn(n, DIM, generator=g) * scale
    return x.half().float()


def reference(target: str, x: torch.Tensor) -> torch.Tensor:
    if target == "k":
        return to_llamacpp_order(_prune_rope_pairs(to_hf_order(x), PRUNE_PCT))
    return _apply(x, threshold=0.0, prune_pct=PRUNE_PCT)


def _scores(target: str, x: torch.Tensor) -> torch.Tensor:
    """The magnitudes the methods rank by: pair energy for keys, |x| for values."""
    if target == "k":
        return x[..., 0::2].square() + x[..., 1::2].square()
    return x.abs()


def run(binary: Path, type_name: str, x: torch.Tensor, q: torch.Tensor):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        x.numpy().astype(np.float32).tofile(tmp / "in.f32")
        q.numpy().astype(np.float32).tofile(tmp / "q.f32")
        subprocess.run([str(binary), type_name, tmp / "in.f32", tmp / "q.f32", tmp / "out.f32", tmp / "dots.f32"],
                       check=True)
        out = torch.from_numpy(np.fromfile(tmp / "out.f32", dtype=np.float32)).reshape(x.shape)
        dots = torch.from_numpy(np.fromfile(tmp / "dots.f32", dtype=np.float32))
    return out, dots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bin", type=Path, required=True, help="built vc_roundtrip")
    parser.add_argument("--vectors", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ok = True
    for target, type_name in (("k", "vck50"), ("v", "vcv50")):
        x = sample(args.vectors, args.seed)
        q = sample(args.vectors, args.seed + 1)
        ref = reference(target, x)
        out, dots = run(args.bin, type_name, x, q)

        kept_ref = ref != 0
        kept_out = out != 0
        mask_diff = (kept_ref != kept_out).any(dim=-1)
        value_diff = (out != ref).any(dim=-1)
        print(f"{type_name}: {args.vectors} vectors, kept {kept_out.sum(-1).float().mean():.1f}/{DIM} values")
        # fp16 inputs repeat magnitudes, and torch.topk breaks ties in no fixed order.
        # A different kept set is fine only when it keeps the same multiset of scores.
        score_ref, score_out = _scores(target, ref), _scores(target, out)
        tie_only = (score_ref.sort(-1).values == score_out.sort(-1).values).all(-1)
        print(f"  vectors with a different kept set: {int(mask_diff.sum())}, of which ties only: {int((mask_diff & tie_only).sum())}")
        print(f"  vectors with any value different:  {int(value_diff.sum())}")
        ok &= bool((tie_only | ~value_diff).all())

        if target == "k":
            q16 = q.half().double()
            want = (ref.double() * q16).sum(-1)
            rel = ((dots.double() - want).abs() / want.abs().clamp_min(1e-6))
            print(f"  vec_dot vs reference q.k: max relative error {rel.max():.2e}, median {rel.median():.2e}")
            ok &= bool(rel.median() < 1e-5)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
