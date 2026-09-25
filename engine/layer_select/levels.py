"""Compression rungs a slot must climb in order (cannot skip)."""

from __future__ import annotations

from engine.layer_select.slots import Slot

LEVELS = (25, 50, 75)


def next_level(current: int, levels: tuple[int, ...] = LEVELS) -> int | None:
    """Next rung after ``current`` (0 = uncompressed).

    Increasing rungs are walked by magnitude, so 25 then 50 then 75. A code
    that shrinks, such as 4 bits then 3 then 2 then 1, is walked in the order
    the tuple lists.
    """
    _check_levels(levels)
    if current < 0:
        raise ValueError("current level must be >= 0")
    if all(levels[index] < levels[index + 1] for index in range(len(levels) - 1)):
        for pct in levels:
            if pct > current:
                return int(pct)
        return None
    if current == 0:
        return int(levels[0])
    for index, pct in enumerate(levels):
        if pct == current:
            if index + 1 == len(levels):
                return None
            return int(levels[index + 1])
    raise ValueError(f"level {current} is not one of {list(levels)}")


def mean_compression(assignment: dict[Slot, int], n_slots: int, rungs=None) -> float:
    """Average fraction of each slot that is removed. Unlisted slots count as 0.

    ``rungs`` supplies the fraction for each level. Without them, the level
    number is a percent.
    """
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    total = 0.0
    for pct in assignment.values():
        if int(pct) <= 0:
            continue
        if rungs is None:
            total += int(pct) / 100.0
        else:
            from engine.layer_select.rungs import fraction_of

            total += fraction_of(tuple(rungs), int(pct))
    return total / n_slots


def _check_levels(levels: tuple[int, ...]) -> None:
    if not levels:
        raise ValueError("levels must be non-empty")
    seen: set[int] = set()
    for pct in levels:
        if not isinstance(pct, int) or isinstance(pct, bool) or pct < 1 or pct > 100 or pct in seen:
            raise ValueError("levels must be distinct integers in 1..100")
        seen.add(pct)
