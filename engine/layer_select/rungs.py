"""Rungs a compression method can climb, and the byte fraction each rung removes.

Prune-style methods use 25/50/75 percent. QJL walks 4, 3, 2, then 1 bits, and
uniform quant walks 8 then 4. The level number is that bit width. A method
that is only on or off for a layer has one rung, and the fraction is how much
of that slot the method drops.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rung:
    level: int
    fraction: float


PRUNE = (Rung(25, 0.25), Rung(50, 0.50), Rung(75, 0.75))
_ON_POOL = (Rung(100, 7 / 8),)
# Each pair always keeps one mean. The residual keeps the other (100 - prune_pct)%.
# Fraction removed is prune_pct / 200. At 100 this is pure pooling, half the bytes.
RESIDUAL = (Rung(25, 25 / 200), Rung(50, 50 / 200), Rung(75, 75 / 200), Rung(100, 100 / 200))
_PAIR_POOL = (Rung(100, 0.5),)
# 4:8 sparsify is on or off for a slot. It drops half the slot.
_ON_SPARSIFY = (Rung(50, 0.5),)
# Mildest code first. 4 bits removes 12/16 of a slot; 1 bit removes 15/16.
QJL = (Rung(4, 12 / 16), Rung(3, 13 / 16), Rung(2, 14 / 16), Rung(1, 15 / 16))
# int8 removes half a slot. int4 removes three quarters. Milder code first.
QUANT = (Rung(8, 0.5), Rung(4, 0.75))

# pair_quant keeps each pair's mean at full width and a residual of 8, 4, 2 or 1 bits.
# A pair stores 16 + bits bits per feature pair instead of 32, so a slot loses
# (16 - bits) / 32 of its bytes, ignoring group scales as QUANT does. Pure
# merging (0 bits, no residual) removes half. It is level 100, because level 0
# already means "uncompressed" in an assignment. No residual width exceeds 8.
MERGE_ONLY = 100
PAIR_QUANT = (
    Rung(8, 8 / 32),
    Rung(4, 12 / 32),
    Rung(2, 14 / 32),
    Rung(1, 15 / 32),
    Rung(MERGE_ONLY, 16 / 32),
)

# pair_rank merges the given percent of a layer's pairs, most similar first. A merged pair
# stores one mean, so it removes half of that pair. pair_rank_residual also keeps the 25% of
# features with the largest difference (32 of Llama 3.1's 128) plus a one-bit mask, so a merged
# pair keeps (17 * 128 + 16 * 32) / (32 * 128) = 0.65625 of its bytes and removes 0.34375.
PAIR_RANK = tuple(Rung(pct, pct / 100 * 0.5) for pct in (25, 50, 75, 100))
PAIR_RANK_RESIDUAL = tuple(Rung(pct, pct / 100 * (1 - (17 * 128 + 16 * 32) / (32 * 128))) for pct in (25, 50, 75, 100))

RUNGS = {
    "vector_compress": PRUNE,
    "vector_compress_pair": PRUNE,
    "checksparse_l1": PRUNE,
    "checksparse_row": PRUNE,
    "sparsify_nm": _ON_SPARSIFY,
    "spatial_pool": _ON_POOL,
    "residual_pool": RESIDUAL,
    "pair_pool": _PAIR_POOL,
    "pair_quant": PAIR_QUANT,
    "pair_rank": PAIR_RANK,
    "pair_rank_residual": PAIR_RANK_RESIDUAL,
    "qjl": QJL,
    "quantize": QUANT,
}


def rungs_for(method: str) -> tuple[Rung, ...]:
    found = RUNGS.get(method)
    if found is None:
        known = ", ".join(sorted(RUNGS))
        raise ValueError(f"no sweep rungs for {method!r}. Known methods: {known}")
    return found


def fraction_of(rungs: tuple[Rung, ...], level: int) -> float:
    for rung in rungs:
        if rung.level == level:
            return rung.fraction
    raise ValueError(f"level {level} is not one of {[rung.level for rung in rungs]}")
