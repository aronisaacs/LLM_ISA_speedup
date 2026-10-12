"""Per-window format choice with a bottom-up tree, independently for each head.

Inside an eight-token window each pair takes its cheapest format, each quad
compares its quad formats against its two pairs, and the eight compares against
its two quads. Cost = loss + price * bits / dense window bits. Every KV head
chooses its own tree; the price is shared by the heads of one layer slot.
"""
from dataclasses import dataclass
import torch

from compression_topics.spatial.algorithms.formats import MENUS, TAG_BITS, WINDOW, dense_bits, group_bits
from compression_topics.spatial.algorithms.representation import loss, prepare, reconstruct, unalign


@dataclass(frozen=True)
class Choice:
    segments: tuple  # (window-relative start, size, residual; None = dense pair)
    loss: float
    payload_bits: int

    @property
    def bits(self):
        return self.payload_bits + TAG_BITS * len(self.segments)


@dataclass
class HeadTable:
    rows: list       # per window: {(size, start): [loss per residual in MENUS[size]]}
    payload: dict    # (size, residual or None) -> payload bits
    window_bits: int
    max_group: int

    def choose(self, window, price):
        row = self.rows[window]

        def visit(start, size):
            if size == 2:
                best = Choice(((start, 2, None),), 0., self.payload[(2, None)])
            else:
                left, right = visit(start, size // 2), visit(start + size // 2, size // 2)
                best = Choice(left.segments + right.segments, left.loss + right.loss,
                              left.payload_bits + right.payload_bits)
            cost = best.loss + price * best.bits / self.window_bits
            if size <= self.max_group:
                for residual, value in zip(MENUS[size], row[(size, start)]):
                    option = Choice(((start, size, residual),), value, self.payload[(size, residual)])
                    candidate = option.loss + price * option.bits / self.window_bits
                    if candidate < cost:  # ties keep the smaller groups
                        best, cost = option, candidate
            return best
        return visit(0, WINDOW)


@dataclass
class WindowChoice:
    heads: tuple  # Choice per head

    @property
    def payload_bits(self):
        return sum(c.payload_bits for c in self.heads)

    @property
    def tags(self):
        return sum(len(c.segments) for c in self.heads)

    @property
    def bits(self):
        return sum(c.bits for c in self.heads)

    @property
    def loss(self):
        return sum(c.loss for c in self.heads) / len(self.heads)


@dataclass
class LayerTable:
    heads: list  # HeadTable per KV head

    def __len__(self):
        return len(self.heads[0].rows)

    @property
    def window_bits(self):
        return sum(h.window_bits for h in self.heads)

    def choose(self, window, price):
        return WindowChoice(tuple(h.choose(window, price) for h in self.heads))


def build(tensor, *, target, max_group=8, rope_tables=None, query_energy=None, fixed_residual=None):
    """Loss table and every candidate reconstruction for whole windows of ``tensor``.

    ``tensor`` is [1, heads, tokens, dim] with tokens a multiple of eight, starting
    on a window boundary. ``fixed_residual`` restricts pairs to dense or one count.
    """
    if max_group not in MENUS:
        raise ValueError('max_group must be 2, 4 or 8')
    if fixed_residual is not None and (max_group != 2 or isinstance(fixed_residual, bool)
                                       or fixed_residual not in MENUS[2]):
        raise ValueError('a fixed residual count requires pairs and 0/16/32 entries')
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < 32:
        raise ValueError('need batch-one [1, KV heads, tokens, head_dim >= 32]')
    heads, length, dim = tensor.shape[1:]
    if length % WINDOW or not length:
        raise ValueError('need whole eight-token windows')
    windows = length // WINDOW
    rows = [[{} for _ in range(windows)] for _ in range(heads)]
    payload = {(2, None): dense_bits(2, dim)}
    reconstructions = {}
    energy = query_energy if target == 'k' else None
    for size, menu in MENUS.items():
        if size > max_group:
            continue
        groups = prepare(tensor, target=target, size=size, rope_tables=rope_tables)
        losses = []
        for residual in menu:
            restored, valid = reconstruct(groups, residual, dtype=torch.bfloat16, query_energy=energy)
            value = loss(groups.source, restored, query_energy=energy).mean(3) * size / WINDOW
            if fixed_residual is not None and residual != fixed_residual:
                value = torch.full_like(value, float('inf'))
            losses.append(value.masked_fill(~valid, float('inf'))[0])
            payload[(size, residual)] = group_bits(residual, dim, size)
            flat = unalign(restored.reshape(1, heads, length, dim), groups)
            reconstructions[(size, residual)] = flat.to(tensor.dtype)
        matrix = torch.stack(losses, -1).cpu().tolist()  # [heads][groups][menu]
        for head in range(heads):
            for window in range(windows):
                for offset in range(0, WINDOW, size):
                    rows[head][window][(size, offset)] = matrix[head][(window * WINDOW + offset) // size]
    table = LayerTable([HeadTable(rows[h], payload, dense_bits(WINDOW, dim), max_group) for h in range(heads)])
    return table, reconstructions


def restore(tensor, choices, reconstructions):
    """Write the chosen reconstruction of each head/segment into a copy of ``tensor``."""
    options = list(reconstructions)
    heads, length = tensor.shape[1], tensor.shape[-2]
    codes = torch.zeros(heads, length, dtype=torch.long)
    for window, choice in enumerate(choices):
        for head, head_choice in enumerate(choice.heads):
            for start, size, residual in head_choice.segments:
                if residual is not None:
                    offset = window * WINDOW + start
                    codes[head, offset:offset + size] = options.index((size, residual)) + 1
    codes = codes.to(tensor.device).reshape(1, heads, length, 1)
    out = tensor.clone()
    for code, option in enumerate(options, 1):
        out = torch.where(codes == code, reconstructions[option], out)
    return out
