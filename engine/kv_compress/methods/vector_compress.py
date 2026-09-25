"""Per-scalar magnitude prune ("vector compression").

Unlike checksparse, this does not decide keep/drop for a whole subvector.
Use ``threshold`` for |x| < T, or ``prune_pct`` to zero the weakest percent of
scalars (percentage control / greedy ladder).
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    threshold: float = 0.0,
    prune_pct: int | None = None,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    if prune_pct is not None:
        if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
            raise ValueError("vector_compress prune_pct must be an integer 0..100")
        n = tensor.shape[-1]
        n_drop = (n * prune_pct) // 100
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


METHODS["vector_compress"] = apply
