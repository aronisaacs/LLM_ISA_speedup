"""Intra-layer mixed precision (accuracy stand-in).

Guess, pending advisor: split the last dim into tiles, rank by L1, assign a
bit-width from `bits` according to `pcts` (loudest tiles get bits[0]). Each
tile is then fake-quantized: integer grid of that width, immediately dequantized
back to the original dtype so stock attention still runs. 16-bit is a no-op;
0-bit zeros the tile (same as dropping it).
"""

from __future__ import annotations

import torch

from kv_compress.methods import METHODS


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    tile: int = 8,
    bits=(16, 8, 4),
    pcts=(25, 50, 25),
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("dynamic_precision tile must be a positive integer")
    bit_list = _as_int_list(bits, "bits")
    pct_list = _as_int_list(pcts, "pcts")
    if len(bit_list) != len(pct_list) or not bit_list:
        raise ValueError("dynamic_precision bits and pcts must be non-empty and the same length")
    if any(width < 0 or width > 16 for width in bit_list):
        raise ValueError("dynamic_precision bits must be integers 0..16")
    if any(pct < 0 or pct > 100 for pct in pct_list) or sum(pct_list) != 100:
        raise ValueError("dynamic_precision pcts must be integers 0..100 that sum to 100")
    if tensor.shape[-1] % tile != 0:
        raise ValueError(
            f"dynamic_precision requires the last dimension to be divisible by tile={tile}, "
            f"got {tensor.shape[-1]}"
        )

    n_tiles = tensor.shape[-1] // tile
    tiles = tensor.reshape(*tensor.shape[:-1], n_tiles, tile)
    counts = _tile_counts(n_tiles, pct_list)
    if all(width >= 16 for width, count in zip(bit_list, counts) if count):
        return tensor
    if all(width == 0 for width, count in zip(bit_list, counts) if count):
        return torch.zeros_like(tensor)

    l1 = tiles.abs().sum(dim=-1)
    rank = l1.argsort(dim=-1, descending=True).argsort(dim=-1)
    assigned = torch.full_like(l1, bit_list[-1], dtype=torch.int64)
    cut = 0
    for width, count in zip(bit_list[:-1], counts[:-1]):
        if count:
            assigned = torch.where(
                (rank >= cut) & (rank < cut + count),
                torch.full_like(assigned, width),
                assigned,
            )
        cut += count

    out = tiles.clone()
    for width in dict.fromkeys(bit_list):
        mask = assigned == width
        if not bool(mask.any()):
            continue
        quantized = _qdq_tiles(tiles, width)
        out = torch.where(mask.unsqueeze(-1), quantized, out)
    return out.reshape_as(tensor)


def _as_int_list(value, label: str) -> list[int]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise TypeError(f"dynamic_precision {label} must be a list of integers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"dynamic_precision {label} entries must be integers")
        out.append(item)
    return out


def _tile_counts(n_tiles: int, pcts: list[int]) -> list[int]:
    counts = [(n_tiles * pct) // 100 for pct in pcts[:-1]]
    counts.append(n_tiles - sum(counts))
    return counts


def _qdq_tiles(tiles: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits <= 0:
        return torch.zeros_like(tiles)
    if n_bits >= 16:
        return tiles
    max_q = max((1 << (n_bits - 1)) - 1, 1)
    scale = tiles.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(tiles.dtype).tiny)
    q = (tiles / scale * max_q).round().clamp(-max_q, max_q)
    return q * (scale / max_q)


METHODS["dynamic_precision"] = apply
