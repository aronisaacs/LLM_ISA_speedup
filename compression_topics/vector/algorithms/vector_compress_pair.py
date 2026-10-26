"""Pair-magnitude prune on keys, scalar prune on values.

Keys follow this codebase's half-split RoPE. Feature ``i`` rotates with
feature ``i + head_dim // 2``, and that pair's magnitude ``a² + b²`` is
unchanged by the rotation. ``prune_pct`` zeros the weakest that percent of
pairs, both features together, so the fraction of the key removed matches
the scalar method.

Values are never rotated, so they use the same per-scalar prune as
``vector_compress``.
"""

from __future__ import annotations

import torch

from compression_topics.vector.algorithms.vector_compress import _apply
from engine.kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    prune_pct: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    if target == "k":
        return _prune_rope_pairs(tensor, prune_pct)
    return _apply(tensor, threshold=0.0, prune_pct=prune_pct)


def _prune_rope_pairs(tensor: torch.Tensor, prune_pct: int) -> torch.Tensor:
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
        raise ValueError("vector_compress_pair prune_pct must be an integer 0..100")
    features = tensor.shape[-1]
    if features % 2 != 0:
        raise ValueError(
            f"vector_compress_pair requires an even feature count, got {features}"
        )
    half = features // 2
    n_drop = (half * prune_pct) // 100
    if n_drop == 0:
        return tensor
    if n_drop >= half:
        return torch.zeros_like(tensor)

    first = tensor[..., :half]
    second = tensor[..., half:]
    magnitude = first.square() + second.square()
    drop_indices = magnitude.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(magnitude, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    dropped = ~keep
    return torch.cat((first.masked_fill(dropped, 0), second.masked_fill(dropped, 0)), dim=-1)


METHODS["vector_compress_pair"] = apply
