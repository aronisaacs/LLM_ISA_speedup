"""Keep the m largest-magnitude values in each n-wide tile along the last dim."""

from __future__ import annotations

import torch

from kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    n: int = 8,
    m: int = 4,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("sparsify_nm n must be a positive integer (tile size)")
    if not isinstance(m, int) or isinstance(m, bool) or m < 0 or m > n:
        raise ValueError("sparsify_nm m must be an integer with 0 <= m <= n")
    if tensor.shape[-1] % n != 0:
        raise ValueError(
            f"sparsify_nm requires the last dimension to be divisible by n={n}, "
            f"got {tensor.shape[-1]}"
        )
    if m == n:
        return tensor
    if m == 0:
        return torch.zeros_like(tensor)

    tiles = tensor.reshape(*tensor.shape[:-1], -1, n)
    keep_indices = tiles.abs().topk(m, dim=-1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(tiles, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_indices, True)
    return tiles.masked_fill(~keep_mask, 0).reshape_as(tensor)


METHODS["sparsify_nm"] = apply
