"""Preprocessing, pair reconstruction and reconstruction error.

Each token is split into a unit direction and a norm. A pair stores the mean
direction m = (a + b) / 2 plus, for a residual format, the largest entries of
the half-difference d = (a - b) / 2. Each token stores one bf16 scale, its norm
divided by the norm of its rebuilt direction (m + d or m - d), so a rebuilt
token is scale x (m +- d) and keeps its original length.

Reconstruction error is relative squared error ||k - k_hat||^2 / ||k||^2.
Query-aware error (keys only) weights each dimension by the calibrated query
energy E[q_d^2] and is absolute: sum_d E[q_d^2] (k_hat_d - k_d)^2, an estimate of
E[(q . (k_hat - k))^2]. The same weights then rank which residual entries are kept.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from compression_topics.spatial.algorithms.formats import kept_entries

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Pairs:
    source: torch.Tensor  # [1, heads, pairs, 2, dim], keys or values as stored
    norms: torch.Tensor  # [1, heads, pairs, 2, 1]
    mean: torch.Tensor  # [1, heads, pairs, 1, dim], mean unit direction
    deviation: torch.Tensor  # [1, heads, pairs, 1, dim], half-difference of the unit directions


def prepare(tensor):
    """Pair up a [1, heads, tokens, dim] slice with an even number of tokens."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("spatial compression requires batch-one [1, heads, tokens, dim] tensors")
    length = tensor.shape[-2]
    if length % 2 or not length:
        raise ValueError("slice must hold whole pairs")
    source = tensor.float()
    norms = source.norm(dim=-1, keepdim=True)
    shape = (1, tensor.shape[1], length // 2, 2, tensor.shape[-1])
    unit = torch.nn.functional.normalize(source, dim=-1).reshape(shape)
    mean = unit.mean(-2, keepdim=True)
    deviation = (unit[..., 0, :] - unit[..., 1, :]).unsqueeze(-2) / 2
    return Pairs(source.reshape(shape), norms.reshape(*shape[:-1], 1), mean, deviation)


def reconstruct(pairs, residual, *, dtype, query_energy=None):
    """Rebuild every pair in one format; returns (tokens, valid[1, heads, pairs])."""
    mean = pairs.mean.to(dtype).float()
    kept = torch.zeros_like(pairs.deviation)
    if residual:
        score = pairs.deviation.square()
        if query_energy is not None:
            score = score * query_energy
        index = score.topk(kept_entries(residual, score.shape[-1]), dim=-1).indices
        kept.scatter_(-1, index, pairs.deviation.gather(-1, index))
    kept = kept.to(dtype).float()
    unit = torch.cat((mean + kept, mean - kept), -2)
    size = unit.norm(dim=-1, keepdim=True)
    scale = (pairs.norms / size.clamp_min(1e-12)).to(dtype).float()
    restored = unit * scale
    valid = (pairs.norms.squeeze(-1) > 1e-12).all(-1) & (size.squeeze(-1) > 1e-6).all(-1)
    valid &= torch.isfinite(restored).all(-1).all(-1)
    return restored, valid


def error(source, restored, *, query_energy=None):
    """Per-token reconstruction error [1, heads, pairs, 2]."""
    squared = (restored - source).square()
    if query_energy is None:
        return squared.sum(-1) / source.square().sum(-1).clamp_min(1e-12)
    return (squared * query_energy).sum(-1)


@lru_cache(maxsize=8)
def _weights(path, digest):
    file = Path(path)
    file = file if file.is_absolute() else ROOT / file
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("query weight file changed; recalibration requires a new output directory")
    saved = torch.load(file, map_location="cpu", weights_only=True)
    weights = saved["weights"].float()
    # Average dimensions j and j + R/2, which rotate together, so the weights do not
    # depend on position; with partial rotation only the first R dimensions rotate.
    rotary = saved.get("rotary_dim") or weights.shape[-1]
    half = rotary // 2
    tied = (weights[..., :half] + weights[..., half:rotary]) / 2
    return torch.cat((tied, tied, weights[..., rotary:]), dim=-1)  # [layers, kv_heads, head_dim]


def query_energy(path, digest, layer_idx, tensor):
    """Calibrated E[q^2] per KV head, shaped to broadcast over paired tokens."""
    weights = _weights(str(path), digest)[layer_idx].to(tensor.device)
    if (
        weights.shape != (tensor.shape[1], tensor.shape[-1])
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("query weights do not match the loaded layer/head shape")
    return weights.reshape(1, tensor.shape[1], 1, 1, tensor.shape[-1])
