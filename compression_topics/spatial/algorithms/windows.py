"""Compress each 16-token window of each KV head to the target saving.

Every head of a layer slot gets the same budget in every window: a fixed-size
block of (1 - saving) x the head's dense window bits, plus one format tag per
pair. A head that saves more leaves the rest of its block unused (padding), so
stored bits always count the full block. Nothing carries over between windows.

Inside one head's window, every pair can step from dense towards PR32, PR16 and
PR0, each step saving bits for added reconstruction error. The steps are sorted
by error added per bit saved and taken in that order until the head meets its
budget (:func:`solve`). A head that cannot reach the budget takes every step.
"""

import torch

from compression_topics.spatial.algorithms.formats import (
    RESIDUALS,
    TAG_BITS,
    WINDOW,
    dense_bits,
    pair_bits,
)
from compression_topics.spatial.algorithms.representation import error, prepare, reconstruct

# Formats from least to most saving: PR32, PR16, PR0. Codes: 0 dense, 1 + index into RESIDUALS.
STEP_ORDER = [2, 1, 0]


def candidates(tensor, *, target, query_energy=None):
    """Reconstruction error and reconstruction of every pair in every merged format.

    ``tensor`` is [1, heads, tokens, dim], tokens a multiple of :data:`WINDOW`.
    Returns (errors [heads, pairs, len(RESIDUALS)], infinite where a pair cannot
    be rebuilt; reconstructions {residual: [1, heads, tokens, dim]}).
    Query energy applies to keys only.
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < 32:
        raise ValueError("need batch-one [1, KV heads, tokens, head_dim >= 32]")
    heads, length, dim = tensor.shape[1:]
    if length % WINDOW or not length:
        raise ValueError(f"need whole {WINDOW}-token windows")
    energy = query_energy if target == "k" else None
    pairs = prepare(tensor)
    columns, reconstructions = [], {}
    for residual in RESIDUALS:
        restored, valid = reconstruct(pairs, residual, dtype=torch.bfloat16, query_energy=energy)
        value = error(pairs.source, restored, query_energy=energy).mean(3)
        columns.append(value.masked_fill(~valid, float("inf"))[0])
        reconstructions[residual] = restored.reshape(1, heads, length, dim).to(tensor.dtype)
    return torch.stack(columns, -1), reconstructions


def solve(errors, dim, saving):
    """Format code of every pair: sort each head's merge steps per window and take them
    until the head saves ``saving`` of its dense window bits.

    Each pair moves from dense to its merged formats in steps along its lower convex
    hull of (bits saved, error). A pair's steps always get more expensive, and a
    stable sort keeps a pair's later step after its earlier one.
    Returns codes [heads, windows, WINDOW] (both tokens of a pair share a code).
    """
    # Small per-window bookkeeping: done on the CPU whatever device the cache is on.
    table = errors.to("cpu", torch.float64)  # [heads, pairs, 3]: PR0, PR16, PR32
    heads, count = table.shape[:2]
    windows = count * 2 // WINDOW
    pairs = WINDOW // 2
    dense = dense_bits(2, dim)
    saved = torch.tensor([dense - pair_bits(RESIDUALS[j], dim) for j in STEP_ORDER],
                         dtype=torch.float64)
    cost = table[..., STEP_ORDER]  # [heads, pairs, 3]
    # Lower convex hull from (0 saved, 0 error): repeatedly take the cheapest next point.
    steps_saved, steps_rate, steps_code = [], [], []
    at_saved = torch.zeros(heads, count, dtype=torch.float64)
    at_error = torch.zeros_like(at_saved)
    at_code = torch.zeros(heads, count, dtype=torch.long)
    for _ in STEP_ORDER:
        gain = saved.view(1, 1, -1) - at_saved.unsqueeze(-1)
        rate = torch.where(gain > 0, (cost - at_error.unsqueeze(-1)) / gain.clamp_min(1),
                           torch.full_like(cost, float("inf")))
        best = rate.argmin(-1)  # ties keep the format that saves less
        best_rate = rate.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        ok = torch.isfinite(best_rate)
        new_saved = torch.where(ok, saved[best], at_saved)
        new_error = torch.where(ok, cost.gather(-1, best.unsqueeze(-1)).squeeze(-1), at_error)
        new_code = torch.where(ok, torch.tensor(STEP_ORDER)[best] + 1, at_code)
        steps_saved.append(new_saved - at_saved)
        steps_rate.append(torch.where(ok, best_rate, torch.full_like(best_rate, float("inf"))))
        steps_code.append(new_code)
        at_saved, at_error, at_code = new_saved, new_error, new_code

    def per_head_window(x):  # [windows * heads, pairs * 3 steps], pair-then-step order
        x = torch.stack(x, -1).view(heads, windows, pairs, len(STEP_ORDER))
        return x.permute(1, 0, 2, 3).reshape(windows * heads, -1)

    step_saved, step_rate = per_head_window(steps_saved), per_head_window(steps_rate)
    need = torch.as_tensor(saving, dtype=torch.float64) * dense_bits(WINDOW, dim)
    rank = torch.sort(step_rate, dim=1, stable=True).indices
    cumulative = step_saved.gather(1, rank).cumsum(1)
    before = cumulative - step_saved.gather(1, rank)
    take = torch.zeros_like(step_rate, dtype=torch.bool).scatter(
        1, rank, (before < need) & torch.isfinite(step_rate.gather(1, rank)))
    taken = (take.view(windows, heads, pairs, -1) * torch.arange(1, len(STEP_ORDER) + 1)).amax(-1)
    taken = taken.permute(1, 0, 2)  # [heads, windows, pairs]: steps taken per pair
    codes = torch.stack(steps_code, -1).view(heads, windows, pairs, -1)
    pair_code = torch.where(
        taken > 0,
        codes.gather(-1, (taken - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
        torch.zeros(heads, windows, pairs, dtype=torch.long))
    return pair_code.unsqueeze(-1).expand(heads, windows, pairs, 2).reshape(heads, windows, WINDOW)


def block_bits(dim, saving):
    """Bits reserved per head per window: (1 - saving) x dense bits plus the pair tags."""
    return (1 - saving) * dense_bits(WINDOW, dim) + (WINDOW // 2) * TAG_BITS


def compress(tensor, *, target, saving, query_energy=None):
    """Compress the whole windows of ``tensor``.

    Returns (reconstructed tensor, stored bits, pair count per format). Stored bits
    count each head's full block, or the bits it used if that is more (a head that
    cannot reach the budget).
    """
    errors, reconstructions = candidates(tensor, target=target, query_energy=query_energy)
    heads, length, dim = tensor.shape[1:]
    code = solve(errors, dim, saving)
    bits_of = torch.tensor([dense_bits(2, dim)] + [pair_bits(r, dim) for r in RESIDUALS],
                           dtype=torch.float64) + TAG_BITS
    used = bits_of[code[..., ::2]].sum(-1)  # [heads, windows]
    block = torch.tensor(block_bits(dim, saving), dtype=torch.float64)
    bits = torch.maximum(used, block).sum(0).sum()  # per window, then over windows
    code = code.reshape(1, heads, length, 1).to(tensor.device)
    out = tensor.clone()
    counts = {"format_dense_pair": int((code == 0).sum()) // 2}
    for k, residual in enumerate(RESIDUALS, 1):
        chosen = code == k
        out = torch.where(chosen, reconstructions[residual], out)
        tokens = int(chosen.sum())
        if tokens:
            counts[f"format_g2_r{residual}"] = tokens // 2
    return out, int(bits), counts
