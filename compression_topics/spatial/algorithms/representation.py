"""Preprocessing, merged-group reconstruction and loss.

Keys are rotated back by their offset inside the group, so every member shares
the frame of the group's first token; values have no RoPE. Each token is split
into a unit direction and a norm. A group stores the mean direction plus, for a
residual format, the largest entries of each deviation from the mean. The
rebuilt direction is rescaled to the token's stored norm.

Loss is relative squared error ||k - k_hat||^2 / ||k||^2. Query-aware loss (keys
only) weights each dimension by the calibrated query energy E[q_d^2] and is
absolute: sum_d E[q_d^2] (k_hat_d - k_d)^2, an estimate of E[(q . (k_hat - k))^2].
The same weights then rank which residual entries are kept.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from engine.kv_compress.rope import apply_rope

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Groups:
    source: torch.Tensor  # [1, heads, groups, size, dim], RoPE-aligned keys or values
    norms: torch.Tensor  # [1, heads, groups, size, 1]
    mean: torch.Tensor  # [1, heads, groups, 1, dim], mean unit direction
    deviation: (
        torch.Tensor
    )  # pairs: [.., 1, dim] half-difference; larger groups: [.., size, dim]
    size: int
    cos: object
    sin: object


def prepare(tensor, *, target, size, rope_tables):
    """Group a [1, heads, tokens, dim] slice whose length is a multiple of ``size``."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "spatial compression requires batch-one [1, heads, tokens, dim] tensors"
        )
    length = tensor.shape[-2]
    if length % size or not length:
        raise ValueError("slice must hold whole groups")
    source = tensor.float()
    cos = sin = None
    if target == "k":
        if rope_tables is None or rope_tables.attention_scaling != 1.0:
            raise ValueError("keys require pure-rotation RoPE tables")
        offsets = torch.arange(length, device=tensor.device) % size
        cos, sin = rope_tables.cos_sin(offsets, torch.float32)
        source = apply_rope(source, cos, sin, inverse=True)
    elif target != "v":
        raise ValueError("target must be 'k' or 'v'")
    norms = source.norm(dim=-1, keepdim=True)
    shape = (1, tensor.shape[1], length // size, size, tensor.shape[-1])
    unit = torch.nn.functional.normalize(source, dim=-1).reshape(shape)
    mean = unit.mean(-2, keepdim=True)
    deviation = (
        (unit[..., 0, :] - unit[..., 1, :]).unsqueeze(-2) / 2
        if size == 2
        else unit - mean
    )
    return Groups(
        source.reshape(shape),
        norms.reshape(*shape[:-1], 1),
        mean,
        deviation,
        size,
        cos,
        sin,
    )


def reconstruct(groups, residual, *, dtype, query_energy=None):
    """Rebuild every group in one format; returns (tokens, valid[1, heads, groups])."""
    mean = groups.mean.to(dtype).float()
    kept = torch.zeros_like(groups.deviation)
    if residual:
        score = groups.deviation.square()
        if query_energy is not None:
            score = score * query_energy
        index = score.topk(residual, dim=-1).indices
        kept.scatter_(-1, index, groups.deviation.gather(-1, index))
    kept = kept.to(dtype).float()
    unit = (
        torch.cat((mean + kept, mean - kept), -2) if groups.size == 2 else mean + kept
    )
    size = unit.norm(dim=-1, keepdim=True)
    scale = (groups.norms / size.clamp_min(1e-12)).to(dtype).float()
    restored = unit * scale
    valid = (groups.norms.squeeze(-1) > 1e-12).all(-1) & (size.squeeze(-1) > 1e-6).all(
        -1
    )
    valid &= torch.isfinite(restored).all(-1).all(-1)
    return restored, valid


def loss(source, restored, *, query_energy=None):
    """Per-token loss [1, heads, groups, size]."""
    squared = (restored - source).square()
    if query_energy is None:
        return squared.sum(-1) / source.square().sum(-1).clamp_min(1e-12)
    return (squared * query_energy).sum(-1)


def unalign(tokens, groups):
    """Undo the key alignment of a [1, heads, tokens, dim] reconstruction."""
    if groups.cos is None:
        return tokens
    return apply_rope(tokens, groups.cos, groups.sin, inverse=False)


@lru_cache(maxsize=8)
def _weights(path, digest):
    file = Path(path)
    file = file if file.is_absolute() else ROOT / file
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(
            "query weight file changed; recalibration requires a new output directory"
        )
    weights = torch.load(file, map_location="cpu", weights_only=True)["weights"].float()
    # Average each RoPE plane (j, j + D/2) so the weights do not depend on rotation.
    half = weights.shape[-1] // 2
    tied = (weights[..., :half] + weights[..., half:]) / 2
    return torch.cat((tied, tied), dim=-1)  # [layers, kv_heads, head_dim]


def query_energy(path, digest, layer_idx, tensor):
    """Calibrated E[q^2] per KV head, shaped to broadcast over grouped tokens."""
    weights = _weights(str(path), digest)[layer_idx].to(tensor.device)
    if (
        weights.shape != (tensor.shape[1], tensor.shape[-1])
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("query weights do not match the loaded layer/head shape")
    return weights.reshape(1, tensor.shape[1], 1, 1, tensor.shape[-1])
