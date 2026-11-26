"""Per-scalar magnitude prune ("vector compression").

Unlike checksparse, this does not decide keep/drop for a whole subvector.
Use ``threshold`` for |x| < T, or ``prune_pct`` to zero the weakest percent of
scalars (percentage control / greedy ladder). ``recent_window`` (tokens) is read by
the cache: the newest tokens stay exact until they leave the window.

``rescale`` (scale check, 2026-11-26): "none" (default, the kept values as they are) or "length"
(the kept values times ||x|| / ||x_kept||, so the stored vector keeps its original length; folded
into the stored values, so it costs no storage); "length_k" / "length_v" rescale only keys / values.

``query_weights`` + ``query_weights_sha256`` (query-aware keys, 2026-11-26; first tried 2026-10, dropped
2026-10-26): keys keep the entries with the largest E[q_d^2] k_d^2 instead of the largest k_d^2, with the
model's calibrated query energy per layer and KV head (the spatial loader, RoPE planes averaged, so the
weights do not depend on rotation). Values have no query side and keep their plain ranking. Same storage.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS


RESCALES = ("none", "length", "length_k", "length_v")


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    threshold: float = 0.0,
    prune_pct: int | float | None = None,
    recent_window: int = 0,
    rescale: str = "none",
    query_weights: str | None = None,
    query_weights_sha256: str | None = None,
    **_unused,
) -> torch.Tensor:
    del _unused, recent_window  # the window is the cache's job
    if rescale not in RESCALES:
        raise ValueError(f"vector_compress rescale must be one of {RESCALES}")
    if query_weights is not None and target == "k" and prune_pct is not None:
        out = _query_keys(tensor, layer_idx, prune_pct, query_weights, query_weights_sha256)
    else:
        out = _apply(tensor, threshold=threshold, prune_pct=prune_pct)
    if rescale == "none" or out is tensor or rescale not in ("length", f"length_{target}"):
        return out
    full = tensor.float().norm(dim=-1, keepdim=True)
    kept = out.float().norm(dim=-1, keepdim=True)
    factor = torch.where(kept > 0, full / kept.clamp_min(1e-12), torch.ones_like(kept))
    return (out.float() * factor).to(tensor.dtype)


def _query_keys(keys, layer_idx, prune_pct, path, digest):
    """Zero the prune_pct of each key's entries with the smallest E[q_d^2] k_d^2."""
    from compression_topics.spatial.algorithms.representation import query_energy

    n = keys.shape[-1]
    n_drop = int(n * prune_pct // 100)
    if n_drop == 0:
        return keys
    weights = query_energy(path, digest, layer_idx, keys[:1]).reshape(1, keys.shape[1], 1, n)
    score = keys.float().square() * weights
    drop = score.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    return keys.scatter(-1, drop, 0)


def _apply(
    tensor: torch.Tensor,
    *,
    threshold: float,
    prune_pct: int | float | None,
) -> torch.Tensor:
    if prune_pct is not None:
        # Whole or fractional percent (37.5 / 62.5 / 87.5: eighths of the vector; Aron, 2026-11-23).
        if not isinstance(prune_pct, (int, float)) or isinstance(prune_pct, bool) or not 0 <= prune_pct <= 100:
            raise ValueError("vector_compress prune_pct must be a number 0..100")
        n = tensor.shape[-1]
        n_drop = int(n * prune_pct // 100)  # whole percents round as before; eighths are exact
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


def _validate(options: dict) -> None:
    window = options.get("recent_window", 0)
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("vector_compress recent_window must be a non-negative integer")
    if (options.get("query_weights") is None) != (options.get("query_weights_sha256") is None):
        raise ValueError("vector_compress query weights need their checksum, and the checksum its weights")
    if options.get("rescale", "none") not in RESCALES:
        raise ValueError(f"vector_compress rescale must be one of {RESCALES}")


METHODS["vector_compress"] = apply
OPTION_VALIDATORS["vector_compress"] = _validate
