"""4:8 activation sparsity on a K or V chunk.

Along the last dimension (head_dim), keep the 4 largest-magnitude values in
each tile of 8 and zero the rest. Same math as Mustafa's _apply_4_to_8_sparsity,
called from Cache.update rather than from modeling_llama.py.

The ratio is fixed. A slot is either 4:8 sparse or untouched, so a step takes
no n or m.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS

N = 8
M = 4


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    seq_start=None,
    rope_tables=None,
    **unexpected,
) -> torch.Tensor:
    del layer_idx, target, seq_start, rope_tables
    if unexpected:
        raise ValueError(f"sparsify_nm is fixed at 4:8 and takes no options, got {sorted(unexpected)}")
    if tensor.shape[-1] % N != 0:
        raise ValueError(
            f"sparsify_nm requires the last dimension to be divisible by {N}, got {tensor.shape[-1]}"
        )
    tiles = tensor.reshape(*tensor.shape[:-1], -1, N)
    keep_indices = tiles.abs().topk(M, dim=-1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(tiles, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_indices, True)
    return tiles.masked_fill(~keep_mask, 0).reshape_as(tensor)


METHODS["sparsify_nm"] = apply
