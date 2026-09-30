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

from compression_topics.vector.algorithms.vector_compress import _apply, _zero_run_starts
from engine.kv_compress.methods import METHODS

_enabled = False
_runs: torch.Tensor | None = None
_runs_done = 0
_vectors = 0
_mask_length: int | None = None


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
        if _enabled:
            _record_mask(torch.ones(*tensor.shape[:-1], half, device=tensor.device, dtype=tensor.dtype))
        return tensor
    if n_drop >= half:
        if _enabled:
            _record_mask(torch.zeros(*tensor.shape[:-1], half, device=tensor.device, dtype=tensor.dtype))
        return torch.zeros_like(tensor)

    first = tensor[..., :half]
    second = tensor[..., half:]
    magnitude = first.square() + second.square()
    drop_indices = magnitude.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(magnitude, dtype=torch.bool)
    keep.scatter_(-1, drop_indices, False)
    if _enabled:
        _record_mask(keep.to(dtype=tensor.dtype))
    dropped = ~keep
    return torch.cat((first.masked_fill(dropped, 0), second.masked_fill(dropped, 0)), dim=-1)


def enable_pair_mask_profile() -> None:
    """Start counting zero runs on the key-pair mask of every later ``apply`` call."""
    global _enabled
    _clear()
    _enabled = True


def disable_pair_mask_profile() -> None:
    """Stop counting and drop any totals still held."""
    global _enabled
    _enabled = False
    _clear()


def take_pair_mask_profile() -> dict:
    """Return ``runs``, ``vectors``, ``mean``, and ``mask_length``, then clear.

    Counting stays on. A zero run is a streak of dropped pairs on the half-length
    mask, not on the reconstructed key. ``mean`` is ``None`` when nothing was recorded.
    """
    runs = _runs_done
    if _runs is not None:
        runs += int(_runs.item())
    vectors = _vectors
    mask_length = _mask_length
    _clear()
    mean = (runs / vectors) if vectors else None
    return {"runs": runs, "vectors": vectors, "mean": mean, "mask_length": mask_length}


def random_pair_mask_runs(mask_length: int, prune_pct: int) -> float:
    """Expected runs if the dropped pairs are a random subset of the mask."""
    dropped = (mask_length * prune_pct) // 100
    return dropped * (mask_length - dropped + 1) / mask_length


def _record_mask(mask: torch.Tensor) -> None:
    """``mask`` is 0 where the pair was dropped and 1 where it was kept."""
    global _runs, _runs_done, _vectors, _mask_length
    if mask.numel() == 0 or mask.shape[-1] == 0:
        return
    length = int(mask.shape[-1])
    if _mask_length is None:
        _mask_length = length
    elif _mask_length != length:
        raise ValueError(f"pair mask length changed from {_mask_length} to {length}")
    with torch.no_grad():
        starts = _zero_run_starts(mask).sum()
    if _runs is None:
        _runs = starts
    elif _runs.device == starts.device:
        _runs = _runs + starts
    else:
        _runs_done += int(_runs.item())
        _runs = starts
    _vectors += mask.numel() // length


def _clear() -> None:
    global _runs, _runs_done, _vectors, _mask_length
    _runs = None
    _runs_done = 0
    _vectors = 0
    _mask_length = None


METHODS["vector_compress_pair"] = apply
