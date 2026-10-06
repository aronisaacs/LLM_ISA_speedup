"""Adjacent-token pooling with a quantized residual. Stock attention still runs on full-width tensors.

Every pair of tokens (0, 1), (2, 3), ... stores the pair mean at full width and
the half-difference ``delta = (x0 - x1) / 2`` quantized to ``bits`` bits. The
two tokens read back as ``mean + delta_q`` and ``mean - delta_q``. An odd tail
token stays exact until the next token arrives and closes the pair.

``bits`` 8 and 4 use the symmetric absmax scheme of ``quantize``: one scale per
group of 32 features, codes in ``-(2**(bits-1) - 1) .. 2**(bits-1) - 1``.
``bits`` 2 has four levels, -1, -1/3, 1/3 and 1 times the group's absmax (a
zero level would waste one of only four codes and does worse than one bit).
``bits`` 1 keeps only the sign of each residual feature, scaled by the mean
absolute residual of its group. ``bits`` 0 stores no residual: both tokens
read back as the mean, which is pure ``pair_pool``.
Group scales and the full-width mean's own storage are not counted by the rung.
Keys are not RoPE aligned.
"""

from __future__ import annotations

import torch

from compression_topics.quantize.algorithms.quantize import _quantize_last
from engine.kv_compress.methods import METHODS

BITS = (8, 4, 2, 1, 0)


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
        raise ValueError(f"pair_quant target must be 'k' or 'v', got {target!r}")
    # A chunk that begins on the second token of a pair is closed later, on the full cache.
    if seq_start % 2 != 0:
        return tensor
    return close_pairs(tensor, bits=bits, group=group)


def write_closed_pairs(
    tensor: torch.Tensor,
    *,
    bits: int,
    group: int,
    start: int,
    end: int,
) -> None:
    """Pool ``tensor[..., start:end, :]`` in place. The span is complete pairs."""
    if end <= start:
        return
    span = tensor[..., start:end, :]
    tensor[..., start:end, :] = close_pairs(span, bits=bits, group=group)


def close_pairs(tensor: torch.Tensor, *, bits: int, group: int = 32) -> torch.Tensor:
    """Pool adjacent tokens along the sequence. A short tail is left unchanged."""
    _check(bits, group)
    sequence = tensor.shape[-2]
    full = (sequence // 2) * 2
    if full == 0:
        return tensor
    body = tensor[..., :full, :]
    first = body[..., 0::2, :]
    second = body[..., 1::2, :]
    mean = (first + second) / 2
    delta = quantize_residual((first - second) / 2, bits, group)
    restored = body.clone()
    restored[..., 0::2, :] = mean + delta
    restored[..., 1::2, :] = mean - delta
    if full == sequence:
        return restored
    return torch.cat((restored, tensor[..., full:, :]), dim=-2)


def quantize_residual(delta: torch.Tensor, bits: int, group: int = 32) -> torch.Tensor:
    """Quantize and restore the residual along the feature axis."""
    if bits == 0:
        return torch.zeros_like(delta)
    if bits == 1:
        return _by_group(delta, group, _sign_group)
    if bits == 2:
        return _by_group(delta, group, _four_levels)
    return _quantize_last(delta, bits, group)


def _by_group(delta: torch.Tensor, group: int, restore) -> torch.Tensor:
    """Apply ``restore`` to each group of ``group`` features. A short last group stands alone."""
    length = delta.shape[-1]
    full = length // group
    pieces = []
    if full:
        body = delta[..., : full * group].reshape(*delta.shape[:-1], full, group)
        pieces.append(restore(body).reshape(*delta.shape[:-1], full * group))
    if length != full * group:
        pieces.append(restore(delta[..., full * group :]))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


def _sign_group(groups: torch.Tensor) -> torch.Tensor:
    """One bit per feature: sign times the group's mean absolute residual."""
    values = groups.float()
    scale = values.abs().mean(dim=-1, keepdim=True)
    signs = torch.where(values >= 0, 1.0, -1.0)
    return (signs * scale).to(dtype=groups.dtype)


def _four_levels(groups: torch.Tensor) -> torch.Tensor:
    """Two bits per feature: codes 0..3 map to -1, -1/3, 1/3, 1 times the group absmax."""
    values = groups.float()
    scale = values.abs().amax(dim=-1, keepdim=True)
    safe = scale.clamp_min(torch.finfo(torch.float32).tiny)
    codes = ((values / safe * 3 + 3) / 2).round().clamp(0, 3)
    restored = torch.where(scale == 0, torch.zeros_like(values), (2 * codes - 3) / 3 * scale)
    return restored.to(dtype=groups.dtype)


def _check(bits: int, group: int) -> None:
    if bits not in BITS:
        raise ValueError(f"pair_quant bits must be one of {list(BITS)}")
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise ValueError("pair_quant group must be a positive integer")


METHODS["pair_quant"] = apply
