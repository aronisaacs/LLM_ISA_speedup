"""Rungs a compression method can climb, and the byte fraction each rung removes.

Prune-style methods use 25/50/75 percent. A method that is only on or off for a
layer has one rung, and the fraction is how much of that slot the method drops.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rung:
    level: int
    fraction: float


PRUNE = (Rung(25, 0.25), Rung(50, 0.50), Rung(75, 0.75))
_ON_POOL = (Rung(100, 7 / 8),)
_ON_TWO_VECTORS = (Rung(100, 6 / 8),)
_ON_FOUR_BIT = (Rung(100, 0.75),)

RUNGS = {
    "vector_compress": PRUNE,
    "checksparse_l1": PRUNE,
    "sparsify_nm": PRUNE,
    "spatial_pool": _ON_POOL,
    "spatial_top1": _ON_TWO_VECTORS,
    "spatial_tile": _ON_TWO_VECTORS,
    "spatial_feature": _ON_TWO_VECTORS,
    "qjl": _ON_FOUR_BIT,
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
