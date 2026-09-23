"""Compression rungs a slot must climb in order (cannot skip)."""

from __future__ import annotations

from layer_select.slots import Slot

LEVELS = (25, 50, 75)


def next_level(current: int, levels: tuple[int, ...] = LEVELS) -> int | None:
    """Next rung strictly above ``current`` (0 = uncompressed)."""
    _check_levels(levels)
    if current < 0:
        raise ValueError("current level must be >= 0")
    for pct in levels:
        if pct > current:
            return int(pct)
    return None


def mean_compression(assignment: dict[Slot, int], n_slots: int) -> float:
    """Average fraction of each slot that is zeroed. Unlisted slots count as 0."""
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    total = sum(max(int(pct), 0) for pct in assignment.values())
    return total / (n_slots * 100.0)


def _check_levels(levels: tuple[int, ...]) -> None:
    if not levels:
        raise ValueError("levels must be non-empty")
    previous = 0
    for pct in levels:
        if not isinstance(pct, int) or isinstance(pct, bool) or pct <= previous or pct > 100:
            raise ValueError("levels must be increasing integers in 1..100")
        previous = pct
