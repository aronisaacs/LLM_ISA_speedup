"""Static chunk rewrites. Same-shape output, so stock attention still runs.

Keys are chunked along the sequence. Values are chunked along the feature
axis: the same rewrite, with those two axes swapped. A tail shorter than
``chunk`` stays exact. The mean is the average of the closed chunk.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply_pool(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    chunk: int = 8,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    return _apply(tensor, target=target, chunk=chunk, kind="pool")


def _apply(tensor: torch.Tensor, *, target: str, chunk: int, kind: str) -> torch.Tensor:
    if target == "v":
        return _rewrite(tensor.transpose(-1, -2), chunk=chunk, kind=kind).transpose(-1, -2)
    return _rewrite(tensor, chunk=chunk, kind=kind)


def _rewrite(tensor: torch.Tensor, *, chunk: int, kind: str) -> torch.Tensor:
    _check_chunk(chunk)
    if tensor.ndim < 2:
        raise ValueError(f"spatial {kind} expects a sequence and a feature dimension")
    sequence = tensor.shape[-2]
    closed = (sequence // chunk) * chunk
    if closed == 0:
        return tensor
    prefix = tensor[..., :closed, :]
    chunks = prefix.reshape(*prefix.shape[:-2], closed // chunk, chunk, prefix.shape[-1])
    mean = chunks.mean(dim=-2, keepdim=True)
    if kind == "pool":
        filled = mean.expand_as(chunks)
    else:
        raise ValueError(f"unknown spatial kind {kind!r}")
    out = tensor.clone()
    out[..., :closed, :] = filled.reshape_as(prefix)
    return out


def _check_chunk(chunk: int) -> None:
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("spatial chunk must be a positive integer")


METHODS["spatial_pool"] = apply_pool
