"""Climb one compression rung at a time from the singleton sweep table.

Each step may raise any layer's key or value to the next of 25%, 50%, 75%.
ΔPPL for a rung is ppl(slot, next) - ppl(slot, current) from the leave-one-out
WikiText scores. Pick argmax (Δcompression / ΔPPL). A slot cannot skip a rung.
"""

from __future__ import annotations

from layer_select.levels import LEVELS, mean_compression, next_level
from layer_select.scores import ScoreRow, score_table
from layer_select.slots import Slot, all_slots


def rank_fill(
    rows: list[ScoreRow],
    n_layers: int,
    budget: float,
    dense_ppl: float | None = None,
) -> dict[Slot, int]:
    slots = all_slots(n_layers)
    table = score_table(rows)
    if dense_ppl is None:
        dense_ppl = _infer_dense_ppl(rows)
    _require_rungs(table, slots)
    assignment = {slot: 0 for slot in slots}
    while mean_compression(assignment, len(slots)) < budget:
        pick = _best_step(assignment, slots, table, dense_ppl)
        if pick is None:
            break
        slot, new_level = pick
        assignment[slot] = new_level
    return {slot: pct for slot, pct in assignment.items() if pct > 0}


def _best_step(assignment, slots, table, dense_ppl):
    best = None
    for slot in slots:
        current = assignment[slot]
        nxt = next_level(current)
        if nxt is None:
            continue
        extra_pct = nxt - current
        extra_ppl = _ppl(table, dense_ppl, slot, nxt) - _ppl(table, dense_ppl, slot, current)
        efficiency = float("inf") if extra_ppl <= 0 else extra_pct / extra_ppl
        # Higher efficiency wins; then more extra compression; then smaller Slot.
        candidate = (efficiency, extra_pct, _slot_sort_key(slot), slot, nxt)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    return best[3], best[4]


def _slot_sort_key(slot: Slot):
    """Larger is better for the max-tuple; prefer smaller Slot on ties."""
    return (-slot.layer, 0 if slot.target == "k" else -1)


def _ppl(table, dense_ppl, slot, level):
    if level == 0:
        return dense_ppl
    row = table.get((slot, level))
    if row is None:
        raise ValueError(f"missing sweep score for {slot.tag()} at {level}%")
    return row.ppl


def _infer_dense_ppl(rows: list[ScoreRow]) -> float:
    if not rows:
        raise ValueError("no sweep scores")
    return rows[0].ppl - rows[0].delta


def _require_rungs(table, slots):
    missing = []
    for slot in slots:
        for pct in LEVELS:
            if (slot, pct) not in table:
                missing.append(f"{slot.tag()}@p{pct}")
    if missing:
        raise ValueError(f"sweep scores missing rungs: {missing}")
