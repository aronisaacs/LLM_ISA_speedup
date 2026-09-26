"""Uniform integer fake-quant. Stock attention still runs on full-width tensors.

Values are grouped along the feature axis, one token at a time. Keys are grouped
along the sequence, one feature at a time, in the chunk this update hands us.
A group of 32 gets one absmax scale. A short tail is its own group. A decode
step that brings a single key token is a group of one, so that token stays exact.
Post-RoPE. The rung ignores the one fp16 scale.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    bits: int = 8,
    group: int = 32,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    _check(bits, group)
    if target not in {"k", "v"}:
        raise ValueError(f"quantize target must be 'k' or 'v', got {target!r}")
    if target == "k":
        return _quantize_last(tensor.transpose(-1, -2), bits, group).transpose(-1, -2)
    return _quantize_last(tensor, bits, group)


def _quantize_last(tensor: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    length = tensor.shape[-1]
    if length == 0:
        return tensor
    max_q = (1 << (bits - 1)) - 1
    full = length // group
    pieces = []
    if full:
        body = tensor[..., : full * group].reshape(*tensor.shape[:-1], full, group)
        pieces.append(_qdq(body, max_q).reshape(*tensor.shape[:-1], full * group))
    if length != full * group:
        pieces.append(_qdq(tensor[..., full * group :], max_q))
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=-1)


def _qdq(groups: torch.Tensor, max_q: int) -> torch.Tensor:
    original = groups.dtype
    values = groups.float()
    scale = values.abs().amax(dim=-1, keepdim=True)
    safe = scale.clamp_min(torch.finfo(torch.float32).tiny)
    codes = (values / safe * max_q).round().clamp(-max_q, max_q)
    restored = torch.where(scale == 0, torch.zeros_like(values), codes * (scale / max_q))
    return restored.to(dtype=original)


def _check(bits: int, group: int) -> None:
    if bits not in (4, 8):
        raise ValueError("quantize bits must be 4 or 8")
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise ValueError("quantize group must be a positive integer")


METHODS["quantize"] = apply
