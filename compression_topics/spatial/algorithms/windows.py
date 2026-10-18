"""Compress each eight-token window of one layer slot to the target saving.

Every completed window gets the same bit budget, (1 - saving) x its dense bits,
shared by all KV heads of the slot. Inside that budget the heads' formats are
chosen together for the least total reconstruction error, so a head whose neighbours agree can
merge more and leave bits to a head that needs a residual. Nothing carries over
from one window to the next.

With pairs only (``max_group=2``) the choice is a sort: every pair can step
from dense towards PR32, PR16 and PR0, each step saving bits for added error.
All steps of a window are sorted by error added per bit saved and taken in that
order until the window fits (:func:`solve_pairs`).

With quads and eights each head chooses its layout with a bottom-up tree: each
pair takes its best format, each quad compares its quad formats against its two
pairs, and the eight against its two quads, scoring error + price x bits. The
price of each window is bisected to the smallest value whose formats fit the
budget; heads that jump to smaller formats at exactly that price then use the
bits left over (:func:`solve`). For pairs both give the same formats.

A window that cannot reach the budget takes its most compressed formats.
"""

import torch

from compression_topics.spatial.algorithms.formats import (
    MENUS,
    TAG_BITS,
    WINDOW,
    dense_bits,
    group_bits,
    sizes,
)
from compression_topics.spatial.algorithms.representation import (
    loss,
    prepare,
    reconstruct,
    unalign,
)

MAX_PRICE = 1e12
STEPS = 60


def candidates(tensor, *, target, max_group=8, rope_tables=None, query_energy=None,
               align_rope=True, keep_norms=True):
    """Loss of every format and every candidate reconstruction of whole windows.

    ``tensor`` is [1, heads, tokens, dim], tokens a multiple of eight. Returns
    (losses, reconstructions): losses[size] is [heads, groups, len(MENUS[size])]
    (infinite where a group cannot be rebuilt); reconstructions[(size, residual)]
    is a full [1, heads, tokens, dim] tensor.
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < 32:
        raise ValueError("need batch-one [1, KV heads, tokens, head_dim >= 32]")
    heads, length, dim = tensor.shape[1:]
    if length % WINDOW or not length:
        raise ValueError("need whole eight-token windows")
    energy = query_energy if target == "k" else None
    losses, reconstructions = {}, {}
    for size in sizes(max_group):
        groups = prepare(tensor, target=target, size=size, rope_tables=rope_tables,
                         align_rope=align_rope, keep_norms=keep_norms)
        columns = []
        for residual in MENUS[size]:
            restored, valid = reconstruct(
                groups, residual, dtype=torch.bfloat16, query_energy=energy
            )
            value = loss(groups.source, restored, query_energy=energy).mean(3) * size / WINDOW
            columns.append(value.masked_fill(~valid, float("inf"))[0])
            flat = unalign(restored.reshape(1, heads, length, dim), groups)
            reconstructions[(size, residual)] = flat.to(tensor.dtype)
        losses[size] = torch.stack(columns, -1)
    return losses, reconstructions


def options(max_group):
    """Merged formats in code order: code k > 0 means ``options[k - 1]``, 0 is dense."""
    return [(s, r) for s in sizes(max_group) for r in MENUS[s]]


def _tree(losses, dim, price, *, codes=False, norms=True):
    """Each head's best layout per window at ``price`` [windows].

    Returns (bits [heads, windows], loss [heads, windows], code [heads, windows, 8]
    or None).
    """
    heads, groups = losses[2].shape[:2]
    windows, device = groups * 2 // WINDOW, losses[2].device
    window_bits = dense_bits(WINDOW, dim)
    scale = price.to(device, torch.float64).view(1, windows, 1) / window_bits
    bits = torch.full(
        (heads, windows, WINDOW // 2),
        float(dense_bits(2, dim) + TAG_BITS),
        dtype=torch.float64,
        device=device,
    )
    error = torch.zeros_like(bits)
    code = torch.zeros(heads, windows, WINDOW // 2, 2, dtype=torch.long, device=device)
    number = 0
    for size in (2, 4, 8):
        n = WINDOW // size
        if size > 2:  # start from the two halves of each group
            bits = bits.view(heads, windows, n, 2).sum(-1)
            error = error.view(heads, windows, n, 2).sum(-1)
            code = code.reshape(heads, windows, n, size)
        if size not in losses:
            continue
        cost = error + scale * bits
        table = losses[size].to(torch.float64).view(heads, windows, n, -1)
        for j, residual in enumerate(MENUS[size]):
            number += 1
            option_bits = group_bits(residual, dim, size, norms) + TAG_BITS
            candidate = table[..., j] + scale * option_bits
            better = candidate < cost  # ties keep the smaller groups
            cost = torch.where(better, candidate, cost)
            bits = torch.where(better, torch.full_like(bits, option_bits), bits)
            error = torch.where(better, table[..., j], error)
            if codes:
                code = torch.where(better.unsqueeze(-1), torch.full_like(code, number), code)
    code = code.reshape(heads, windows, WINDOW) if codes else None
    return bits.view(heads, windows), error.view(heads, windows), code


def solve(losses, dim, saving, norms=True):
    """Formats of every window within its budget; returns (bits per window, codes)."""
    heads, groups = losses[2].shape[:2]
    windows = groups * 2 // WINDOW
    limit = (1 - saving) * heads * dense_bits(WINDOW, dim)

    def fits(price):
        return _tree(losses, dim, price, norms=norms)[0].sum(0).cpu() <= limit

    # Bisect log10(price) per window: ``hi`` fits the budget, ``lo`` does not.
    lo = torch.full((windows,), -9.0, dtype=torch.float64)
    hi = torch.full((windows,), 12.0, dtype=torch.float64)
    for _ in range(STEPS):
        middle = (lo + hi) / 2
        ok = fits(10**middle)
        hi, lo = torch.where(ok, middle, hi), torch.where(ok, lo, middle)
    free = fits(torch.zeros(windows, dtype=torch.float64))
    high = torch.where(free, 0.0, 10**hi)
    low = torch.where(free, 0.0, 10**lo)
    bits_hi, loss_hi, code_hi = _tree(losses, dim, high, codes=True, norms=norms)
    bits_lo, loss_lo, code_lo = _tree(losses, dim, low, codes=True, norms=norms)
    # Between the two prices some heads jump to smaller formats. Undo the jumps
    # that save the most loss per bit while the window stays within its budget.
    extra = (bits_lo - bits_hi).cpu()
    gain = (loss_hi - loss_lo).cpu()
    rate = torch.where(extra > 0, gain / extra.clamp_min(1), torch.zeros_like(gain))
    order = rate.argsort(0, descending=True)
    spent = extra.gather(0, order).cumsum(0)
    slack = limit - bits_hi.sum(0).cpu()
    take = torch.zeros_like(extra, dtype=torch.bool).scatter(0, order, spent <= slack)
    take = (take & (extra > 0)).to(code_hi.device)
    bits = torch.where(take, bits_lo, bits_hi)
    code = torch.where(take.unsqueeze(-1), code_lo, code_hi)
    return bits.sum(0).cpu(), code


def solve_pairs(losses, dim, saving, norms=True, per_head=False, window=WINDOW,
                tags_in_budget=True):
    """Pairs only: sort the merge steps of a window and take them until it fits.

    Each pair can move from dense to its merged formats in steps along its lower
    convex hull of (bits saved, error): every step saves some bits for some extra
    error. All steps of a window, over all heads, are sorted by error added per
    bit saved and taken in that order until the window meets its budget (or every
    step is taken). With ``per_head`` every head of the window must meet the
    saving on its own instead of sharing the window's budget. Returns (bits per
    window, codes), as :func:`solve`. ``window`` is the number of tokens decided
    together (8 by default; 16 waits for twice as many tokens). With ``per_head``,
    ``saving`` may also be one saving per KV head (each head keeps its own fixed
    share of the slot's budget).
    """
    # Small per-window bookkeeping: done on the CPU whatever device the cache is on.
    table = losses[2].to("cpu", torch.float64)  # [heads, pairs, 3]: PR0, PR16, PR32
    heads, pairs = table.shape[:2]
    windows = pairs * 2 // window
    dense = dense_bits(2, dim)
    # Formats from least to most saving: PR32, PR16, PR0 (codes 3, 2, 1).
    order = [2, 1, 0]
    saved = torch.tensor([dense - group_bits(MENUS[2][j], dim, 2, norms) for j in order], dtype=torch.float64)
    error = table[..., order]  # [heads, pairs, 3]
    # Lower convex hull from (0 saved, 0 error): repeatedly take the cheapest next point.
    steps_saved, steps_error, steps_code = [], [], []
    at_saved = torch.zeros(heads, pairs, dtype=torch.float64)
    at_error = torch.zeros_like(at_saved)
    at_code = torch.zeros(heads, pairs, dtype=torch.long)
    for _ in range(len(order)):
        gain = saved.view(1, 1, -1) - at_saved.unsqueeze(-1)
        rate = torch.where(gain > 0, (error - at_error.unsqueeze(-1)) / gain.clamp_min(1),
                           torch.full_like(error, float("inf")))
        best = rate.argmin(-1)  # ties keep the format that saves less
        best_rate = rate.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        ok = torch.isfinite(best_rate)
        new_saved = torch.where(ok, saved[best], at_saved)
        new_error = torch.where(ok, error.gather(-1, best.unsqueeze(-1)).squeeze(-1), at_error)
        new_code = torch.where(ok, torch.tensor(order)[best] + 1, at_code)
        steps_saved.append(new_saved - at_saved)
        steps_error.append(torch.where(ok, best_rate, torch.full_like(best_rate, float("inf"))))
        steps_code.append(new_code)
        at_saved, at_error, at_code = new_saved, new_error, new_code
    # [windows, heads * 4 pairs * 3 steps], in pair-then-step order so a stable sort
    # never takes a pair's later step before its earlier one.
    def per_window(x):
        x = torch.stack(x, -1).view(heads, windows, window // 2, len(order))
        return x.permute(1, 0, 2, 3).reshape(windows * (heads if per_head else 1), -1)

    step_saved, step_rate, step_code = map(per_window, (steps_saved, steps_error, steps_code))
    sharing = 1 if per_head else heads  # heads that share one budget
    target = torch.as_tensor(saving, dtype=torch.float64)
    if target.ndim:  # one saving per head: groups are ordered (window, head)
        if not per_head or target.shape != (heads,):
            raise ValueError("per-head savings need per_head and one value per KV head")
        target = target.repeat(windows).unsqueeze(1)
    tags = (window // 2) * TAG_BITS if tags_in_budget else 0  # padded heads reserve tags apart
    need = sharing * (target * dense_bits(window, dim) + tags)
    rank = torch.sort(step_rate, dim=1, stable=True).indices
    cumulative = step_saved.gather(1, rank).cumsum(1)
    before = cumulative - step_saved.gather(1, rank)
    take = torch.zeros_like(step_rate, dtype=torch.bool).scatter(
        1, rank, (before < need) & torch.isfinite(step_rate.gather(1, rank)))
    taken = (take.view(windows, heads, window // 2, -1) * torch.arange(1, len(order) + 1)).amax(-1)
    codes = torch.stack(steps_code, -1).view(heads, windows, window // 2, -1)
    pair_code = torch.where(
        taken.permute(1, 0, 2) > 0,
        codes.gather(-1, (taken.permute(1, 0, 2) - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
        torch.zeros(heads, windows, window // 2, dtype=torch.long))
    bits_of = torch.tensor([dense] + [group_bits(r, dim, 2, norms) for r in MENUS[2]], dtype=torch.float64) + TAG_BITS
    bits = bits_of[pair_code].sum((0, 2))
    code = pair_code.unsqueeze(-1).expand(heads, windows, window // 2, 2).reshape(heads, windows, window)
    return bits, code


def head_shares(losses, dim, saving, norms=True):
    """Saving each KV head reaches when the heads share each window's budget.

    Returns (sum over windows of each head's saving, number of windows), in the
    same units as a per-head ``saving`` for :func:`solve_pairs`.
    """
    _, code = solve_pairs(losses, dim, saving, norms)
    bits_of = torch.tensor(
        [dense_bits(2, dim)] + [group_bits(r, dim, 2, norms) for r in MENUS[2]],
        dtype=torch.float64) + TAG_BITS
    per_window = bits_of[code[..., ::2].cpu()].sum(-1)  # [heads, windows]
    pairs = WINDOW // 2
    # A head meets a per-head saving s when its bits, tags included, are at most
    # (1 - s) x its dense bits plus its tags (see solve_pairs).
    saving = (pairs * dense_bits(2, dim) - (per_window - pairs * TAG_BITS)) / dense_bits(WINDOW, dim)
    return saving.sum(1), saving.shape[1]


def round_to_levels(shares, saving, levels):
    """Round per-head savings to ``levels`` so the heads still average at least ``saving``.

    Every head first takes the largest level at or below its share; then the heads
    that were rounded down the most move up one level until the average reaches
    ``saving`` (or every head is at the top level).
    """
    levels = sorted(levels)
    index = [max(i for i, v in enumerate(levels) if v <= s + 1e-12) for s in shares]
    while sum(levels[i] for i in index) / len(index) < saving - 1e-12:
        movable = [h for h, i in enumerate(index) if i + 1 < len(levels)]
        if not movable:
            break
        h = max(movable, key=lambda h: shares[h] - levels[index[h]])
        index[h] += 1
    return [levels[i] for i in index]


def compress(tensor, *, target, saving, max_group=8, rope_tables=None, query_energy=None,
             align_rope=True, keep_norms=True, per_head_budget=False, window=WINDOW,
             pad_heads=False):
    """Compress the whole windows of ``tensor``.

    Returns (reconstructed tensor, stored bits, segment count per format). With
    ``pad_heads`` (per-head savings), every head reserves exactly its budget in every
    window, so stored bits count the reserved size rather than the bits used.
    """
    losses, reconstructions = candidates(
        tensor,
        target=target,
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=query_energy,
        align_rope=align_rope,
        keep_norms=keep_norms,
    )
    heads, length, dim = tensor.shape[1:]
    if max_group == 2:
        bits, code = solve_pairs(losses, dim, saving, keep_norms, per_head_budget, window,
                                 tags_in_budget=not pad_heads)
    else:
        bits, code = solve(losses, dim, saving, keep_norms)
    if pad_heads:
        bits_of = torch.tensor(
            [dense_bits(2, dim)] + [group_bits(r, dim, 2, keep_norms) for r in MENUS[2]],
            dtype=torch.float64) + TAG_BITS
        used = bits_of[code[..., ::2].cpu()].sum(-1)  # [heads, windows]
        share = torch.as_tensor(saving, dtype=torch.float64).view(-1, 1)
        reserved = (1 - share) * dense_bits(window, dim) + (window // 2) * TAG_BITS
        bits = torch.maximum(used, reserved).sum(0)
    code = code.reshape(1, heads, length, 1).to(tensor.device)
    out = tensor.clone()
    counts = {"format_dense_pair": int((code == 0).sum()) // 2}
    for k, (size, residual) in enumerate(options(max_group), 1):
        chosen = code == k
        out = torch.where(chosen, reconstructions[(size, residual)], out)
        tokens = int(chosen.sum())
        if tokens:
            counts[f"format_g{size}_r{residual}"] = tokens // size
    return out, int(bits.sum()), counts
