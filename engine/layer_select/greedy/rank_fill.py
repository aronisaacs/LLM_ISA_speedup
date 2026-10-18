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
    saving=None,
    layers=None,
) -> dict[Slot, int]:
    """Climb rungs until the mean compression of the ``targets`` slots reaches ``budget``.

    ``saving(slot, rung)`` is the fraction of the slot a rung saves, if it is not
    the rung's own fraction (for example after metadata overhead). ``layers`` limits
    the slots to the layers that hold a KV cache, when not all do (Qwen3.5).
    """
    slots = all_slots(n_layers, targets)
    if layers is not None:
        slots = tuple(slot for slot in slots if slot.layer in set(layers))
    table = score_table(rows)
    if dense_ppl is None:
        dense_ppl = _infer_dense_ppl(rows)
    _require_rungs(table, slots, rungs)
    from engine.layer_select.greedy.calibrated import choose_candidates
    choices = {(slot.layer, slot.target): [
        {"level": rung.level, "compression": rung.fraction,
         "allocation_saving": rung.fraction if saving is None else saving(slot, rung),
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
