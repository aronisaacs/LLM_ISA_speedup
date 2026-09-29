"""Per-scalar magnitude prune ("vector compression").

Unlike checksparse, this does not decide keep/drop for a whole subvector.
Use ``threshold`` for |x| < T, or ``prune_pct`` to zero the weakest percent of
scalars (percentage control / greedy ladder).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.kv_compress.methods import METHODS

_enabled = False
_runs: torch.Tensor | None = None
_runs_done = 0
_vectors = 0


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
    result = _apply(tensor, threshold=threshold, prune_pct=prune_pct)
    if _enabled:
        _record(result)
    return result


def zero_run_count(tensor: torch.Tensor) -> int:
    """Maximal exact-zero streaks on the last dimension, summed over every vector."""
    if tensor.numel() == 0 or tensor.shape[-1] == 0:
        return 0
    return int(_zero_run_starts(tensor).sum().item())


def enable_zero_run_profile() -> None:
    """Start counting zero runs on every later ``apply`` call."""
    global _enabled
    _clear()
    _enabled = True


def disable_zero_run_profile() -> None:
    """Stop counting and drop any totals still held."""
    global _enabled
    _enabled = False
    _clear()


def take_zero_run_profile() -> dict:
    """Return ``runs``, ``vectors``, and ``mean``, then clear the totals.

    Counting stays on, so the next ``apply`` calls start a fresh average.
    ``mean`` is ``None`` when nothing has been recorded.
    """
    runs = _runs_done
    if _runs is not None:
        runs += int(_runs.item())
    vectors = _vectors
    _clear()
    mean = (runs / vectors) if vectors else None
    return {"runs": runs, "vectors": vectors, "mean": mean}


def _apply(
    tensor: torch.Tensor,
    *,
    threshold: float,
    prune_pct: int | None,
) -> torch.Tensor:
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


def _record(tensor: torch.Tensor) -> None:
    global _runs, _runs_done, _vectors
    if tensor.numel() == 0 or tensor.shape[-1] == 0:
        return
    with torch.no_grad():
        starts = _zero_run_starts(tensor).sum()
    if _runs is None:
        _runs = starts
    elif _runs.device == starts.device:
        _runs = _runs + starts
    else:
        _runs_done += int(_runs.item())
        _runs = starts
    _vectors += tensor.numel() // tensor.shape[-1]


def _zero_run_starts(tensor: torch.Tensor) -> torch.Tensor:
    zero = tensor == 0
    previous_zero = F.pad(zero[..., :-1], (1, 0), value=False)
    return zero & ~previous_zero


def _clear() -> None:
    global _runs, _runs_done, _vectors
    _runs = None
    _runs_done = 0
    _vectors = 0


METHODS["vector_compress"] = apply
