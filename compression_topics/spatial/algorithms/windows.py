"""Compress each eight-token window of one layer slot to the target saving.

Every completed window gets the same bit budget, (1 - saving) x its dense bits,
shared by all KV heads of the slot. Inside that budget the heads' formats are
chosen together for the least total loss, so a head whose neighbours agree can
merge more and leave bits to a head that needs a residual. Nothing carries over
from one window to the next.

Each head chooses its layout with a bottom-up tree: each pair takes its best
format (dense, or merged with 0/16/32 residual entries), each quad compares its
quad formats against its two pairs, and the eight against its two quads, scoring
loss + price x bits. The price of each window is bisected to the smallest value
whose formats fit the budget; heads that jump to smaller formats at exactly that
price then use the bits left over. A window that cannot reach the budget takes
its most compressed formats.
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


def candidates(tensor, *, target, max_group=8, rope_tables=None, query_energy=None):
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
        groups = prepare(tensor, target=target, size=size, rope_tables=rope_tables)
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


def _tree(losses, dim, price, *, codes=False):
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
            option_bits = group_bits(residual, dim, size) + TAG_BITS
            candidate = table[..., j] + scale * option_bits
            better = candidate < cost  # ties keep the smaller groups
            cost = torch.where(better, candidate, cost)
            bits = torch.where(better, torch.full_like(bits, option_bits), bits)
            error = torch.where(better, table[..., j], error)
            if codes:
                code = torch.where(better.unsqueeze(-1), torch.full_like(code, number), code)
    code = code.reshape(heads, windows, WINDOW) if codes else None
    return bits.view(heads, windows), error.view(heads, windows), code


def solve(losses, dim, saving):
    """Formats of every window within its budget; returns (bits per window, codes)."""
    heads, groups = losses[2].shape[:2]
    windows = groups * 2 // WINDOW
    limit = (1 - saving) * heads * dense_bits(WINDOW, dim)

    def fits(price):
        return _tree(losses, dim, price)[0].sum(0).cpu() <= limit

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
    bits_hi, loss_hi, code_hi = _tree(losses, dim, high, codes=True)
    bits_lo, loss_lo, code_lo = _tree(losses, dim, low, codes=True)
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


def compress(tensor, *, target, saving, max_group=8, rope_tables=None, query_energy=None):
    """Compress the whole windows of ``tensor``.

    Returns (reconstructed tensor, stored bits, segment count per format).
    """
    losses, reconstructions = candidates(
        tensor,
        target=target,
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=query_energy,
    )
    heads, length, dim = tensor.shape[1:]
    bits, code = solve(losses, dim, saving)
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
