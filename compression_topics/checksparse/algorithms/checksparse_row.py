"""Zero the weakest L1 tiles across every KV head of a token (Mustafa's checksparse).

For each token, the K or V heads are laid side by side into one row of
kv_heads * head_dim features (1024 on Llama 3.1 8B) and split into tiles of
`tile`. Each tile is scored by L1, summed in the input dtype as he does, and
the lowest `prune_pct` percent of the row's tiles are zeroed. A weak head can
lose most of its tiles while a strong head keeps all of its own.

checksparse_l1 drops the same share inside every head instead. On the same
slot this row-wide choice costs less WikiText perplexity, more so at 75%.
Like his `_apply_tile_sparsity`, any prune_pct above 0 drops at least one tile.
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
        raise ValueError("checksparse_row tile must be a positive integer")
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or prune_pct < 0 or prune_pct > 100:
        raise ValueError("checksparse_row prune_pct must be an integer 0..100")
    if tensor.dim() != 4:
        raise ValueError(f"checksparse_row expects [batch, kv_heads, seq, head_dim], got {tuple(tensor.shape)}")
    if prune_pct == 0:
        return tensor

    batch, heads, seq, head_dim = tensor.shape
    width = heads * head_dim
    if width % tile != 0:
        raise ValueError(
            f"checksparse_row requires kv_heads * head_dim divisible by tile={tile}, got {width}"
        )
    n_tiles = width // tile
    n_drop = min(max((n_tiles * prune_pct) // 100, 1), n_tiles)

    tiles = tensor.transpose(1, 2).reshape(batch, seq, n_tiles, tile)
    l1 = tiles.abs().sum(dim=-1).to(torch.float32)
    drop_indices = l1.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(l1, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    pruned = tiles.masked_fill(~keep.unsqueeze(-1), 0)
    return pruned.reshape(batch, seq, heads, head_dim).transpose(1, 2).contiguous()


METHODS["checksparse_row"] = apply
