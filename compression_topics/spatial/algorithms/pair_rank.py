"""Merge the most similar adjacent token pairs of a layer, a fixed share of them. Stock attention still runs on full-width tensors.

Every pair of tokens (0, 1), (2, 3), ... in a layer's keys (or values) is scored by its relative error
|rest| / |m|, with m = (x0 + x1) / 2 and d = (x0 - x1) / 2. ``rest`` is d with the ``keep_pct``
percent of features of largest |d| removed, because those are stored exactly. The ``pct`` percent
of pairs with the smallest score are merged: both tokens read back as m + kept and m - kept. The
other pairs stay exact. Pairs are ranked across all KV heads of the sequence, so ``pct`` is the
share of the layer's head-pairs that merge, whatever their error.

``pair_rank`` keeps no residual (a merged pair costs half). ``pair_rank_residual`` keeps the
top 1/4 of features of d exactly (a merged pair costs (17 D + 16 D / 4) / (32 D), 0.656).

This acts on a whole prefill pass only: the share is a property of the full sequence, so a chunk
that starts after position 0 (generation) is left exact. Rank also needs one real sequence per
call, because padding would rank as the most similar pairs, so a batch larger than 1 is refused.

``rope`` (keys only) aligns the second token of each pair into the first token's RoPE frame by
one step before the mean, and rotates the reconstruction back.

Counts of pairs seen and merged go into ``pair_gate.STATS`` per target, so a run reports its
actual stored bytes (see ``pair_gate.bytes_vs_dense``).
"""

from __future__ import annotations

import torch

from compression_topics.spatial.algorithms import pair_gate
from engine.kv_compress.methods import METHODS
from engine.kv_compress.rope import RopeTables, apply_rope

RESIDUAL_KEEP_PCT = 25


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    pct: int = 50,
    keep_pct: int = 0,
    rope: bool = False,
    seq_start: int = 0,
    rope_tables: RopeTables | None = None,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    if target not in {"k", "v"}:
        raise ValueError(f"pair_rank target must be 'k' or 'v', got {target!r}")
    _check(pct, keep_pct, rope)
    if seq_start != 0:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("pair_rank needs batch size 1: padded positions would rank as the most similar pairs")
    return merge_top_pairs(
        tensor,
        pct=pct,
        keep_pct=keep_pct,
        rope=rope and target == "k",
        rope_tables=rope_tables,
        target=target,
    )


def apply_residual(tensor: torch.Tensor, *, keep_pct: int = RESIDUAL_KEEP_PCT, **kwargs) -> torch.Tensor:
    """``pair_rank`` with the top 1/4 of the difference kept exactly."""
    del keep_pct
    return apply(tensor, keep_pct=RESIDUAL_KEEP_PCT, **kwargs)


def merge_top_pairs(
    tensor: torch.Tensor,
    *,
    pct: int,
    keep_pct: int,
    rope: bool = False,
    rope_tables: RopeTables | None = None,
    target: str | None = None,
) -> torch.Tensor:
    """Merge the ``pct`` percent of pairs with the smallest score. A short tail is left unchanged."""
    _check(pct, keep_pct, rope)
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
    delta = (first - second) / 2
    kept = pair_gate._largest(delta, (delta.shape[-1] * keep_pct) // 100)
    score = (delta - kept).float().norm(dim=-1) / mean.float().norm(dim=-1).clamp_min(1e-8)
    merge = _smallest(score, pct).unsqueeze(-1)
    merged_second = mean - kept
    if rope:
        merged_second = _shift_rope(merged_second, rope_tables, inverse=False)
    restored = body.clone()
    restored[..., 0::2, :] = torch.where(merge, mean + kept, body[..., 0::2, :])
    restored[..., 1::2, :] = torch.where(merge, merged_second, body[..., 1::2, :])
    if target is not None:
        pair_gate._count(target, merge.numel(), int(merge.sum()), delta.shape[-1], (delta.shape[-1] * keep_pct) // 100)
    if full == sequence:
        return restored
    return torch.cat((restored, tensor[..., full:, :]), dim=-2)


def _smallest(score: torch.Tensor, pct: int) -> torch.Tensor:
    """Mask of the ``pct`` percent smallest scores, ranked within each batch element over heads and pairs."""
    flat = score.reshape(score.shape[0], -1)
    count = (flat.shape[-1] * pct + 50) // 100
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count > 0:
        index = flat.topk(count, dim=-1, largest=False, sorted=False).indices
        mask.scatter_(-1, index, True)
    return mask.reshape(score.shape)


def _shift_rope(tokens: torch.Tensor, rope_tables: RopeTables | None, *, inverse: bool) -> torch.Tensor:
    if rope_tables is None:
        raise ValueError("pair_rank rope alignment requires rope tables")
    positions = torch.ones(tokens.shape[-2], device=tokens.device, dtype=torch.float32)
    cos, sin = rope_tables.cos_sin(positions, tokens.dtype)
    return apply_rope(tokens, cos, sin, inverse=inverse)


def _check(pct: int, keep_pct: int, rope: bool) -> None:
    for name, value in (("pct", pct), ("keep_pct", keep_pct)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 100:
            raise ValueError(f"pair_rank {name} must be an integer 0..100")
    if not isinstance(rope, bool):
        raise ValueError("pair_rank rope must be a bool")


METHODS["pair_rank"] = apply
METHODS["pair_rank_residual"] = apply_residual
