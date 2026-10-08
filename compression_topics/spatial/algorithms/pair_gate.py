"""Gated adjacent-token pooling with a sparse exact residual. Stock attention still runs on full-width tensors.

Every pair of tokens (0, 1), (2, 3), ... is looked at. With m = (x0 + x1) / 2 and
d = (x0 - x1) / 2, the ``keep_pct`` percent of features with the largest |d| are stored
exactly as a residual, and ``rest`` is what remains of d. The pair is merged when
|rest| / |m| <= ``tau``. A merged pair reads back as m + kept and m - kept. A pair that fails
the test stays exact, both tokens unchanged. ``keep_pct`` 0 keeps no residual (plain merge of
similar pairs). An odd tail token stays exact until the next token arrives and closes the pair.

Stored bytes per pair, as a share of two dense vectors, are 1/2 when nothing is kept and
(16 D + D + 16 k) / (32 D) when k of D features are kept (mean, a one-bit-per-feature mask,
and the kept differences). Unmerged pairs cost 1. Per-pair flags are not counted.

Counts of pairs seen and merged are accumulated in ``STATS`` per target (k or v), so a run can
report its actual bytes. ``reset_stats`` and ``pop_stats`` read and clear them.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS, AFTER_APPEND

from engine.kv_compress.metrics import STATS, reset_stats, pop_stats


def _count(target: str, pairs: int, merged: int, features: int, kept: int) -> None:
    entry = STATS.setdefault(target, {"pairs": 0, "merged": 0, "features": features, "kept": kept})
    entry["pairs"] += pairs
    entry["merged"] += merged


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    tau: float = 0.3,
    keep_pct: int = 0,
    seq_start: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    _check(tau, keep_pct)
    if target not in {"k", "v"}:
        raise ValueError(f"pair_gate target must be 'k' or 'v', got {target!r}")
    # A chunk that begins on the second token of a pair is closed later, on the full cache.
    if seq_start % 2 != 0:
        return tensor
    return close_pairs(tensor, tau=tau, keep_pct=keep_pct, target=target)


def write_closed_pairs(
    tensor: torch.Tensor,
    *,
    tau: float,
    keep_pct: int,
    target: str,
    start: int,
    end: int,
) -> None:
    """Gate and merge ``tensor[..., start:end, :]`` in place. The span is complete pairs."""
    if end <= start:
        return
    span = tensor[..., start:end, :]
    tensor[..., start:end, :] = close_pairs(span, tau=tau, keep_pct=keep_pct, target=target)


def close_pairs(tensor: torch.Tensor, *, tau: float, keep_pct: int, target: str | None = None) -> torch.Tensor:
    """Merge the adjacent pairs that pass the gate. A short tail is left unchanged."""
    _check(tau, keep_pct)
    sequence = tensor.shape[-2]
    full = (sequence // 2) * 2
    if full == 0:
        return tensor
    body = tensor[..., :full, :]
    first = body[..., 0::2, :]
    second = body[..., 1::2, :]
    mean = (first + second) / 2
    delta = (first - second) / 2
    kept = _largest(delta, (delta.shape[-1] * keep_pct) // 100)
    rest = (delta - kept).float()
    relative = rest.norm(dim=-1) / mean.float().norm(dim=-1).clamp_min(1e-8)
    merge = (relative <= tau).unsqueeze(-1)
    restored = body.clone()
    restored[..., 0::2, :] = torch.where(merge, mean + kept, first)
    restored[..., 1::2, :] = torch.where(merge, mean - kept, second)
    if target is not None:
        _count(target, merge.numel(), int(merge.sum()), delta.shape[-1], (delta.shape[-1] * keep_pct) // 100)
    if full == sequence:
        return restored
    return torch.cat((restored, tensor[..., full:, :]), dim=-2)


from compression_topics.spatial.algorithms.residual import largest_residual as _largest


def bytes_vs_dense(stats: dict[str, int]) -> float:
    """Stored bytes of the gated slots as a share of dense, from the counts in ``STATS``."""
    pairs, merged = stats["pairs"], stats["merged"]
    if pairs == 0:
        return 1.0
    features, kept = stats["features"], stats["kept"]
    merged_cost = 0.5 if kept == 0 else (17 * features + 16 * kept) / (32 * features)
    return (merged * merged_cost + (pairs - merged)) / pairs


def _check(tau: float, keep_pct: int) -> None:
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or tau < 0:
        raise ValueError("pair_gate tau must be a number >= 0")
    if not isinstance(keep_pct, int) or isinstance(keep_pct, bool) or keep_pct < 0 or keep_pct > 100:
        raise ValueError("pair_gate keep_pct must be an integer 0..100")


METHODS["pair_gate"] = apply


def after_append(tensor, *, target, layer_idx, start, end, rope_tables=None, **options):
    if start % 2:
        write_closed_pairs(tensor, tau=options.get("tau", .3), keep_pct=options.get("keep_pct", 0),
                           target=target, start=start // 2 * 2, end=end // 2 * 2)


AFTER_APPEND["pair_gate"] = after_append
