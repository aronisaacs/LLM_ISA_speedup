"""Per-scalar magnitude prune ("vector compression").

Unlike checksparse, this does not decide keep/drop for a whole subvector.
Use ``threshold`` for |x| < T, or ``prune_pct`` to zero the weakest percent of
scalars (percentage control / greedy ladder). ``recent_window`` (tokens) is read by
the cache: the newest tokens stay exact until they leave the window.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    threshold: float = 0.0,
    prune_pct: int | float | None = None,
    recent_window: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused, recent_window  # the window is the cache's job
    return _apply(tensor, threshold=threshold, prune_pct=prune_pct)


def _apply(
    tensor: torch.Tensor,
    *,
    threshold: float,
    prune_pct: int | float | None,
) -> torch.Tensor:
    if prune_pct is not None:
        # Whole or fractional percent (37.5 / 62.5 / 87.5: eighths of the vector; Aron, 2026-11-23).
        if not isinstance(prune_pct, (int, float)) or isinstance(prune_pct, bool) or not 0 <= prune_pct <= 100:
            raise ValueError("vector_compress prune_pct must be a number 0..100")
        n = tensor.shape[-1]
        n_drop = int(n * prune_pct // 100)  # whole percents round as before; eighths are exact
        if n_drop == 0:
            return tensor
        if n_drop >= n:
            return torch.zeros_like(tensor)
        drop_indices = tensor.abs().topk(n_drop, dim=-1, largest=False, sorted=False).indices
        keep = torch.ones_like(tensor, dtype=torch.bool)
        keep.scatter_(-1, drop_indices, False)
        return tensor.masked_fill(~keep, 0)

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("vector_compress threshold must be a real number")
    if threshold < 0:
        raise ValueError("vector_compress threshold must be >= 0")
    if threshold == 0:
        return tensor
    return tensor.masked_fill(tensor.abs() < threshold, 0)


def _validate(options: dict) -> None:
    window = options.get("recent_window", 0)
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("vector_compress recent_window must be a non-negative integer")


METHODS["vector_compress"] = apply
OPTION_VALIDATORS["vector_compress"] = _validate
