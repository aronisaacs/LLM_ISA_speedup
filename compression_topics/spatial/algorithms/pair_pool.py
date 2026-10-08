"""Pure adjacent-token pooling. No residual, and no RoPE alignment.

Every included layer replaces each token pair with its mean. This is the same
math as ``residual_pool`` at ``prune_pct`` 100 with rope off, registered on its
own so a sweep can only turn a layer on or off.
"""

from __future__ import annotations

import torch

from compression_topics.spatial.algorithms.residual_pool import close_pairs
from engine.kv_compress.methods import METHODS, AFTER_APPEND


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    seq_start: int = 0,
    **_unused,
) -> torch.Tensor:
    del layer_idx, _unused
    if target not in {"k", "v"}:
        raise ValueError(f"pair_pool target must be 'k' or 'v', got {target!r}")
    if seq_start % 2 != 0:
        return tensor
    return close_pairs(tensor, prune_pct=100, rope=False, rope_tables=None)


METHODS["pair_pool"] = apply


def after_append(tensor, *, target, layer_idx, start, end, rope_tables=None, **options):
    from compression_topics.spatial.algorithms.residual_pool import write_closed_pairs
    if start % 2:
        write_closed_pairs(tensor, prune_pct=100, rope=False, rope_tables=None,
                           start=start // 2 * 2, end=end // 2 * 2)


AFTER_APPEND["pair_pool"] = after_append
