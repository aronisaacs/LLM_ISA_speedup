"""Sequential greedy: at each step, try every legal next rung on the current set.

A slot cannot skip a compression level. Needs a live evaluate callback.
"""

from __future__ import annotations

from collections.abc import Callable

from layer_select.levels import DEFAULT_LEVELS, mean_compression, next_level
from layer_select.slots import Slot

EvaluateAssignment = Callable[[dict[Slot, int]], float]


def run_sequential(
    *,
    slots: tuple[Slot, ...],
    budget: float,
    evaluate_assignment: EvaluateAssignment,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    dense_ppl: float | None = None,
) -> dict[Slot, int]:
    if dense_ppl is None:
        dense_ppl = evaluate_assignment({})
    assignment = {slot: 0 for slot in slots}
    current_ppl = dense_ppl
    while mean_compression(assignment, len(slots)) < budget:
        best = None
        for slot in slots:
            nxt = next_level(assignment[slot], levels)
            if nxt is None:
                continue
            trial = {item: pct for item, pct in assignment.items() if pct > 0}
            trial[slot] = nxt
            extra_pct = nxt - assignment[slot]
            ppl = evaluate_assignment(trial)
            extra_ppl = ppl - current_ppl
            efficiency = float("inf") if extra_ppl <= 0 else extra_pct / extra_ppl
            candidate = (efficiency, extra_pct, _slot_sort_key(slot), slot, nxt, ppl)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None:
            break
        _eff, _pct, _key, slot, nxt, current_ppl = best
        assignment[slot] = nxt
    return {slot: pct for slot, pct in assignment.items() if pct > 0}


def _slot_sort_key(slot: Slot):
    return (-slot.layer, 0 if slot.target == "k" else -1)
