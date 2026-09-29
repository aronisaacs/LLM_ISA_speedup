"""Zero weakest L1 tiles (Mustafa's PyTorch checksparse stand-in).

Along the last dimension, split into tiles of `tile` (8 or 16 in his slides).
Score each tile by L1 (sum of abs). Drop the lowest `prune_pct` percent of tiles
by zeroing them. Same-shape rewrite: stock attention still runs; this is the
accuracy proxy for a skip bit, not the ISA skip itself.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    tile: int = 8,
    prune_pct: int = 50,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("checksparse_l1 tile must be a positive integer")
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
        raise ValueError("checksparse_l1 prune_pct must be an integer 0..100")
    if tensor.shape[-1] % tile != 0:
        raise ValueError(
            f"checksparse_l1 requires the last dimension to be divisible by tile={tile}, "
            f"got {tensor.shape[-1]}"
        )
    n_tiles = tensor.shape[-1] // tile
    n_drop = (n_tiles * prune_pct) // 100
    if n_drop == 0:
        return tensor
    if n_drop >= n_tiles:
        return torch.zeros_like(tensor)

    tiles = tensor.reshape(*tensor.shape[:-1], n_tiles, tile)
    l1 = tiles.abs().sum(dim=-1)
    drop_indices = l1.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(l1, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    return tiles.masked_fill(~keep.unsqueeze(-1), 0).reshape_as(tensor)


METHODS["checksparse_l1"] = apply
