"""Pair-magnitude prune, a variant of vector compression.

Features are grouped as adjacent pairs (2i, 2i+1). A pair's magnitude is
a² + b². ``prune_pct`` zeros the weakest that percent of pairs, both features
together, so the fraction of the vector removed matches the scalar method.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    prune_pct: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
        raise ValueError("vector_compress_pair prune_pct must be an integer 0..100")
    features = tensor.shape[-1]
    if features % 2 != 0:
        raise ValueError(
            f"vector_compress_pair requires an even feature count, got {features}"
        )
    n_pairs = features // 2
    n_drop = (n_pairs * prune_pct) // 100
    if n_drop == 0:
        return tensor
    if n_drop >= n_pairs:
        return torch.zeros_like(tensor)

    pairs = tensor.reshape(*tensor.shape[:-1], n_pairs, 2)
    magnitude = pairs.square().sum(dim=-1)
    drop_indices = magnitude.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(magnitude, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    return pairs.masked_fill(~keep.unsqueeze(-1), 0).reshape_as(tensor)


METHODS["vector_compress_pair"] = apply
