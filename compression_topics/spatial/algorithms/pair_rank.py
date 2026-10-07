"""Merge the most similar share of adjacent token pairs in a layer, optionally keeping a residual.

Every pair of tokens (0, 1), (2, 3), ... is looked at in each KV head. With m = (x0 + x1) / 2
and d = (x0 - x1) / 2, the ``keep_pct`` percent of features with the largest |d| are kept
exactly, and ``rest`` is what remains of d. A pair's distance is |rest| / |m|. On the first
chunk of a sequence (prefill) the pairs of each sequence are ranked across all heads of the
layer, and the ``merge_pct`` percent with the smallest distance are merged: a merged pair reads
back as m + kept and m - kept, and every other pair stays exact. ``keep_pct`` 0 is a plain merge.

Pairs that close later (decode) are not ranked. They merge when their distance is at most the
largest distance merged in that layer's last prefill. An odd tail token stays exact until the
next token arrives and closes the pair. Keys are compared as stored, with RoPE applied.

Stored bytes per pair, as a share of two dense vectors, are 1/2 when nothing is kept and
(16 D + D + 16 k) / (32 D) when k of D features are kept (mean, a one-bit-per-feature mask, and
the kept differences). Unmerged pairs cost 1. Per-pair flags are not counted. ``STATS`` holds
the pairs seen and merged per target, which a run reports next to its scores.

``pair_rank`` and ``pair_rank_residual`` are the same method under two names, so each gets its
own compression rungs in a layer sweep; they differ only in the default ``keep_pct``.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS

STATS: dict[str, dict[str, int]] = {}
# Largest distance merged in the last prefill, per (layer, target); used for pairs closed later.
THRESHOLDS: dict[tuple[int, str], float] = {}


def reset_stats() -> None:
    STATS.clear()
    THRESHOLDS.clear()


def pop_stats() -> dict[str, dict[str, int]]:
    """Counts per target since the last reset, then clear them."""
    snapshot = {target: dict(counts) for target, counts in STATS.items()}
    reset_stats()
    return snapshot


def kept_features(features: int, keep_pct: int) -> int:
    return (features * keep_pct) // 100


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    merge_pct: int = 50,
    keep_pct: int = 0,
    seq_start: int = 0,
    **_unused,
) -> torch.Tensor:
    del _unused
    _check(merge_pct, keep_pct)
    if target not in {"k", "v"}:
        raise ValueError(f"pair_rank target must be 'k' or 'v', got {target!r}")
    # A chunk that begins on the second token of a pair is closed later, on the full cache.
    if seq_start % 2 != 0:
        return tensor
    if seq_start == 0:
        return close_pairs(tensor, merge_pct=merge_pct, keep_pct=keep_pct, target=target, layer_idx=layer_idx)
    return close_pairs(tensor, merge_pct=merge_pct, keep_pct=keep_pct, target=target, layer_idx=layer_idx, ranked=False)


def write_closed_pairs(
    tensor: torch.Tensor,
    *,
    merge_pct: int,
    keep_pct: int,
    target: str,
    layer_idx: int,
    start: int,
    end: int,
) -> None:
    """Merge the closed pairs in ``tensor[..., start:end, :]`` in place, against the prefill threshold."""
    if end <= start:
        return
    span = tensor[..., start:end, :]
    tensor[..., start:end, :] = close_pairs(
        span, merge_pct=merge_pct, keep_pct=keep_pct, target=target, layer_idx=layer_idx, ranked=False
    )


def close_pairs(
    tensor: torch.Tensor,
    *,
    merge_pct: int,
    keep_pct: int,
    target: str | None = None,
    layer_idx: int | None = None,
    ranked: bool = True,
) -> torch.Tensor:
    """Merge the chosen adjacent pairs of ``tensor`` [batch, heads, seq, dim]. A short tail is left unchanged.

    ``ranked`` merges exactly the ``merge_pct`` percent closest pairs of each sequence (over heads
    and positions together) and records the threshold; otherwise pairs within the recorded
    threshold of this layer merge.
    """
    _check(merge_pct, keep_pct)
    sequence = tensor.shape[-2]
    full = (sequence // 2) * 2
    if full == 0:
        return tensor
    body = tensor[..., :full, :]
    first = body[..., 0::2, :]
    second = body[..., 1::2, :]
    mean = (first + second) / 2
    delta = (first - second) / 2
    kept = _largest(delta, kept_features(delta.shape[-1], keep_pct))
    rest = (delta - kept).float()
    distance = rest.norm(dim=-1) / mean.float().norm(dim=-1).clamp_min(1e-8)  # [batch, heads, pairs]
    if ranked:
        merge = _closest(distance, merge_pct)
        if layer_idx is not None and target is not None:
            merged = distance[merge]
            THRESHOLDS[(layer_idx, target)] = float(merged.max()) if merged.numel() else float("-inf")
    else:
        threshold = THRESHOLDS.get((layer_idx, target), float("-inf")) if layer_idx is not None else float("-inf")
        merge = distance <= threshold
    restored = body.clone()
    gate = merge.unsqueeze(-1)
    restored[..., 0::2, :] = torch.where(gate, mean + kept, first)
    restored[..., 1::2, :] = torch.where(gate, mean - kept, second)
    if target is not None:
        _count(target, merge.numel(), int(merge.sum()), delta.shape[-1], kept_features(delta.shape[-1], keep_pct))
    if full == sequence:
        return restored
    return torch.cat((restored, tensor[..., full:, :]), dim=-2)


def _closest(distance: torch.Tensor, merge_pct: int) -> torch.Tensor:
    """Boolean mask of the ``merge_pct`` percent smallest distances in each sequence (dim 0)."""
    batch = distance.shape[0]
    flat = distance.reshape(batch, -1)
    count = round(flat.shape[1] * merge_pct / 100)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count > 0:
        order = flat.argsort(dim=1, stable=True)
        mask.scatter_(1, order[:, :count], True)
    return mask.view_as(distance)


def _largest(delta: torch.Tensor, count: int) -> torch.Tensor:
    """The ``count`` largest-|d| entries of each vector, zeros elsewhere."""
    if count <= 0:
        return torch.zeros_like(delta)
    if count >= delta.shape[-1]:
        return delta
    index = delta.abs().topk(count, dim=-1).indices
    return torch.zeros_like(delta).scatter(-1, index, delta.gather(-1, index))


def _count(target: str, pairs: int, merged: int, features: int, kept: int) -> None:
    entry = STATS.setdefault(target, {"pairs": 0, "merged": 0, "features": features, "kept": kept})
    entry["pairs"] += pairs
    entry["merged"] += merged


def merged_pair_cost(features: int, keep_pct: int) -> float:
    """Bytes of a merged pair as a share of the two dense vectors."""
    kept = kept_features(features, keep_pct)
    return 0.5 if kept == 0 else (17 * features + 16 * kept) / (32 * features)


def bytes_vs_dense(stats: dict[str, int]) -> float:
    """Stored bytes of the ranked slots as a share of dense, from the counts in ``STATS``."""
    pairs, merged = stats["pairs"], stats["merged"]
    if pairs == 0:
        return 1.0
    features, kept = stats["features"], stats["kept"]
    cost = 0.5 if kept == 0 else (17 * features + 16 * kept) / (32 * features)  # merged_pair_cost, from a kept count
    return (merged * cost + (pairs - merged)) / pairs


def _check(merge_pct: int, keep_pct: int) -> None:
    for name, value in (("merge_pct", merge_pct), ("keep_pct", keep_pct)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 100:
            raise ValueError(f"pair_rank {name} must be an integer 0..100")


def _apply_residual(tensor: torch.Tensor, *, keep_pct: int = 25, **kwargs) -> torch.Tensor:
    return apply(tensor, keep_pct=keep_pct, **kwargs)


METHODS["pair_rank"] = apply
METHODS["pair_rank_residual"] = _apply_residual
