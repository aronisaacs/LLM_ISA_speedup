"""Adjacent-token residual pool. Stock attention still runs on full-width tensors.

Every pair of tokens (0, 1), (2, 3), ... stores the pair mean on both positions
and a residual on the features with the largest |delta|. ``prune_pct`` is the
fraction of those features whose residual is zeroed. 100 is pure pooling.
An odd tail token stays exact until the next token arrives and closes the pair.

Keys may be aligned into the first token's RoPE frame before the mean and
delta are computed. The partner is rotated backward by one step, then the
reconstructed partner is rotated forward again. Values are never rotated.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS
from engine.kv_compress.rope import RopeTables, apply_rope


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    prune_pct: int = 25,
    rope: bool = False,
    seq_start: int = 0,
    rope_tables: RopeTables | None = None,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    if target not in {"k", "v"}:
        raise ValueError(f"residual_pool target must be 'k' or 'v', got {target!r}")
    if not isinstance(rope, bool):
        raise ValueError("residual_pool rope must be a bool")
    # A chunk that begins on the second token of a pair is closed later, on the full cache.
    if seq_start % 2 != 0:
        return tensor
    align = rope and target == "k"
    return close_pairs(tensor, prune_pct=prune_pct, rope=align, rope_tables=rope_tables)


def write_closed_pairs(
    tensor: torch.Tensor,
    *,
    prune_pct: int,
    rope: bool,
    rope_tables: RopeTables | None,
    start: int,
    end: int,
) -> None:
    """Pool ``tensor[..., start:end, :]`` in place. The span is complete pairs."""
    if end <= start:
        return
    span = tensor[..., start:end, :]
    tensor[..., start:end, :] = close_pairs(
        span, prune_pct=prune_pct, rope=rope, rope_tables=rope_tables
    )


def close_pairs(
    tensor: torch.Tensor,
    *,
    prune_pct: int,
    rope: bool,
    rope_tables: RopeTables | None,
) -> torch.Tensor:
    """Pool adjacent tokens along the sequence. A short tail is left unchanged."""
    _check_prune(prune_pct)
    if not isinstance(rope, bool):
        raise ValueError("residual_pool rope must be a bool")
    sequence = tensor.shape[-2]
    full = (sequence // 2) * 2
    if full == 0:
        return tensor
    body = tensor[..., :full, :]
    first = body[..., 0::2, :]
    second = body[..., 1::2, :]
    if rope:
        second = _shift_rope(second, rope_tables, inverse=True)
    mean = (first + second) / 2
    delta = _keep_residual((first - second) / 2, prune_pct)
    restored_first = mean + delta
    restored_second = mean - delta
    if rope:
        restored_second = _shift_rope(restored_second, rope_tables, inverse=False)
    restored = body.clone()
    restored[..., 0::2, :] = restored_first
    restored[..., 1::2, :] = restored_second
    if full == sequence:
        return restored
    return torch.cat((restored, tensor[..., full:, :]), dim=-2)


def _keep_residual(delta: torch.Tensor, prune_pct: int) -> torch.Tensor:
    features = delta.shape[-1]
    n_drop = (features * prune_pct) // 100
    if n_drop == 0:
        return delta
    if n_drop >= features:
        return torch.zeros_like(delta)
    drop_indices = delta.abs().topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(delta, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    return delta.masked_fill(~keep, 0)


def _shift_rope(tokens: torch.Tensor, rope_tables: RopeTables | None, *, inverse: bool) -> torch.Tensor:
    if rope_tables is None:
        raise ValueError("residual_pool rope alignment requires rope tables")
    count = tokens.shape[-2]
    positions = torch.ones(count, device=tokens.device, dtype=torch.float32)
    cos, sin = rope_tables.cos_sin(positions, tokens.dtype)
    return apply_rope(tokens, cos, sin, inverse=inverse)


def _check_prune(prune_pct: int) -> None:
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
        raise ValueError("residual_pool prune_pct must be an integer 0..100")


METHODS["residual_pool"] = apply
