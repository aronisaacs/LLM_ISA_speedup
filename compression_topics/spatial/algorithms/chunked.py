"""Fixed-ratio chunks: every eight-token window is compressed to the target on its own.

Each completed window of one layer slot gets the same bit budget,
(1 - saving) x its dense bits. Within that budget the heads' formats are chosen
together to minimise the window's total loss, so a head with similar neighbours
can merge more and leave bits to a head that needs a residual. Nothing carries
over from one window to the next: no startup, no feedback, no running price.

The choice inside a window is solved with a per-window multiplier: the bottom-up
tree of ``windows`` is evaluated for all heads and windows at once at a price per
window, and each window's price is bisected to the smallest value whose formats
fit the budget; heads that only jump to smaller formats at that price then use
any bits left over. A window that cannot reach the budget takes its most
compressed formats.
"""

import torch

from compression_topics.spatial.algorithms.formats import TAG_BITS, WINDOW, dense_bits
from compression_topics.spatial.algorithms.windows import candidates

MAX_PRICE = 1e12
STEPS = 60


def _tree(tables, payload, window_bits, price, *, codes=False):
    """Best formats per head and window at ``price`` [windows].

    Returns (bits [heads, windows], loss [heads, windows], code [heads, windows, 8]
    or None). Code 0 is dense; k > 0 is ``options[k - 1]`` of :func:`options`.
    """
    any_table = next(iter(tables.values()))
    heads, device = any_table.shape[0], any_table.device
    windows = any_table.shape[1] * next(iter(tables)) // WINDOW
    scale = price.to(device, torch.float64).view(1, windows, 1) / window_bits
    pairs = WINDOW // 2
    bits = torch.full((heads, windows, pairs), float(payload[(2, None)] + TAG_BITS),
                      dtype=torch.float64, device=device)
    error = torch.zeros_like(bits)
    code = torch.zeros(heads, windows, pairs, 2, dtype=torch.long, device=device) if codes else None
    number = 0
    for size in (2, 4, 8):
        n = WINDOW // size
        if size > 2:  # children: the two halves of each group
            bits = bits.view(heads, windows, n, 2).sum(-1)
            error = error.view(heads, windows, n, 2).sum(-1)
            if codes:
                code = code.reshape(heads, windows, n, size)
        cost = error + scale * bits
        if size not in tables:
            continue
        table = tables[size].to(torch.float64).view(heads, windows, n, -1)
        for j, residual in enumerate(_menu(payload, size)):
            number += 1
            option_bits = payload[(size, residual)] + TAG_BITS
            option_error = table[..., j]
            candidate = option_error + scale * option_bits
            better = candidate < cost  # ties keep the smaller groups
            cost = torch.where(better, candidate, cost)
            bits = torch.where(better, torch.full_like(bits, option_bits), bits)
            error = torch.where(better, option_error, error)
            if codes:
                code = torch.where(better.unsqueeze(-1), torch.full_like(code, number), code)
    return bits.view(heads, windows), error.view(heads, windows), (
        code.reshape(heads, windows, WINDOW) if codes else None)


def _menu(payload, size):
    return [r for (s, r) in payload if s == size and r is not None]


def options(payload):
    """Merged formats in code order (code k is options[k - 1])."""
    return [(s, r) for s in (2, 4, 8) for r in _menu(payload, s)]


def solve(tables, payload, window_bits, heads, saving, *, minimum_storage=False):
    """Per-window formats within the budget; returns (bits per window, codes)."""
    windows = next(iter(tables.values())).shape[1] * next(iter(tables)) // WINDOW
    limit = (1 - saving) * window_bits * heads
    if minimum_storage:
        price = torch.full((windows,), MAX_PRICE, dtype=torch.float64)
        bits, _, code = _tree(tables, payload, window_bits, price, codes=True)
        return bits.sum(0).cpu(), code
    # Bisect log10(price) per window: ``hi`` fits the budget, ``lo`` does not.
    lo = torch.full((windows,), -9.0, dtype=torch.float64)
    hi = torch.full((windows,), 12.0, dtype=torch.float64)
    free = _tree(tables, payload, window_bits, torch.zeros(windows))[0].sum(0).cpu() <= limit
    for _ in range(STEPS):
        middle = (lo + hi) / 2
        ok = _tree(tables, payload, window_bits, 10**middle)[0].sum(0).cpu() <= limit
        hi = torch.where(ok, middle, hi)
        lo = torch.where(ok, lo, middle)
    high = torch.where(free, torch.zeros(windows, dtype=torch.float64), 10**hi)
    low = torch.where(free, high, 10**lo)
    bits_hi, loss_hi, code_hi = _tree(tables, payload, window_bits, high, codes=True)
    bits_lo, loss_lo, code_lo = _tree(tables, payload, window_bits, low, codes=True)
    # Between the two prices some heads jump to more compressed formats. Spend the
    # remaining budget undoing the jumps that save the most loss per bit.
    extra = (bits_lo - bits_hi).cpu()  # [heads, windows], >= 0
    gain = (loss_hi - loss_lo).cpu()
    rate = torch.where(extra > 0, gain / extra.clamp_min(1), torch.zeros_like(gain))
    order = rate.argsort(0, descending=True)
    spent = extra.gather(0, order).cumsum(0)
    slack = limit - bits_hi.sum(0).cpu()
    take = torch.zeros_like(extra, dtype=torch.bool).scatter(0, order, spent <= slack)
    take &= extra > 0
    use_lo = take.to(code_hi.device)
    bits = torch.where(use_lo, bits_lo, bits_hi)
    code = torch.where(use_lo.unsqueeze(-1), code_lo, code_hi)
    return bits.sum(0).cpu(), code


def compress(tensor, *, target, saving, max_group=8, rope_tables=None, query_energy=None,
             fixed_residual=None, residual_counts=None, merge_sizes=None, minimum_storage=False):
    """Compress whole windows of [1, heads, tokens, dim]; returns (tensor, stored bits, format counts)."""
    tables, payload, reconstructions = candidates(
        tensor, target=target, max_group=max_group, rope_tables=rope_tables,
        query_energy=query_energy, fixed_residual=fixed_residual,
        residual_counts=residual_counts, merge_sizes=merge_sizes)
    heads, length, dim = tensor.shape[1:]
    window_bits = dense_bits(WINDOW, dim)
    bits, code = solve(tables, payload, window_bits, heads, saving, minimum_storage=minimum_storage)
    code = code.reshape(1, heads, length, 1).to(tensor.device)
    out = tensor.clone()
    counts = {"format_dense_pair": int((code == 0).sum()) // 2}
    for k, (size, residual) in enumerate(options(payload), 1):
        chosen = code == k
        if residual is not None and (size, residual) in reconstructions:
            out = torch.where(chosen, reconstructions[(size, residual)], out)
        tokens = int(chosen.sum())
        if tokens:
            counts[f"format_g{size}_r{residual}"] = tokens // size
    return out, int(bits.sum()), counts
