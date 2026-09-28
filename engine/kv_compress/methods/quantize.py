"""Uniform integer fake-quant. Stock attention still runs on full-width tensors.

Values are grouped along the feature axis, one token at a time. Keys are grouped
along the sequence, one feature at a time, in groups of 32 that share one absmax
scale. A short tail stays full precision. The cache writes a group back once later
tokens fill it, including keys stored on earlier steps. A single new key stays
exact until its group closes. Post-RoPE. The rung ignores the one fp16 scale.
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
    seq_start: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    _check(bits, group)
    if target not in {"k", "v"}:
        raise ValueError(f"quantize target must be 'k' or 'v', got {target!r}")
    if target == "k":
        # A chunk that begins inside an open group is closed later, on the full cache.
        if seq_start % group != 0:
            return tensor
        return _quantize_closed(tensor.transpose(-1, -2), bits, group).transpose(-1, -2)
    return _quantize_last(tensor, bits, group)


def write_closed_key_groups(
    keys: torch.Tensor,
    *,
    bits: int,
    group: int,
    start: int,
    end: int,
) -> None:
    """Quantize ``keys[..., start:end, :]`` in place. The span is complete groups."""
    if end <= start:
        return
    span = keys[..., start:end, :]
    quantized = _quantize_closed(span.transpose(-1, -2), bits, group).transpose(-1, -2)
    keys[..., start:end, :] = quantized


def _quantize_closed(tensor: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Quantize full groups along the last dim. A short tail is left unchanged."""
    length = tensor.shape[-1]
    full = length // group
    if full == 0:
        return tensor
    max_q = (1 << (bits - 1)) - 1
    width = full * group
    body = tensor[..., :width]
    quantized = _qdq(body.reshape(*tensor.shape[:-1], full, group), max_q).reshape_as(body)
    if width == length:
        return quantized
    return torch.cat((quantized, tensor[..., width:]), dim=-1)


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
