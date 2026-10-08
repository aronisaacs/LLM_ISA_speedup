"""Climb one compression rung at a time from the singleton sweep table.

Each step may raise any layer's key or value to the next of 25%, 50%, 75%.
ΔPPL for a rung is ppl(slot, next) - ppl(slot, current) from the leave-one-out
WikiText scores. Pick argmax (Δcompression / ΔPPL). A slot cannot skip a rung.
"""

from __future__ import annotations

from engine.layer_select.rungs import PRUNE
from engine.layer_select.scores import ScoreRow, score_table
from engine.layer_select.slots import Slot, all_slots


def rank_fill(
    rows: list[ScoreRow],
    n_layers: int,
    budget: float,
    dense_ppl: float | None = None,
    rungs=PRUNE,
    targets: tuple[str, ...] = ("k", "v"),
) -> dict[Slot, int]:
    """Climb rungs until the mean compression of the ``targets`` slots reaches ``budget``."""
    slots = all_slots(n_layers, targets)
    table = score_table(rows)
    if dense_ppl is None:
        dense_ppl = _infer_dense_ppl(rows)
    _require_rungs(table, slots, rungs)
    from engine.layer_select.greedy.calibrated import choose_candidates
    choices = {(slot.layer, slot.target): [
        {"level": rung.level, "compression": rung.fraction,
         "ppl": _ppl(table, dense_ppl, slot, rung.level)} for rung in rungs]
        for slot in slots}
    selected, _ = choose_candidates(choices, dense_ppl, budget, require_budget=False)
    return {Slot(layer, target): row["level"] for (layer, target), row in selected.items()}


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


def _require_rungs(table, slots, rungs):
    missing = []
    for slot in slots:
        for pct in (rung.level for rung in rungs):
            if (slot, pct) not in table:
                missing.append(f"{slot.tag()}@p{pct}")
    if missing:
        raise ValueError(f"sweep scores missing rungs: {missing}")
