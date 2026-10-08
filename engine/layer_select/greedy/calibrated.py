"""One greedy allocator for fixed rungs and measured per-slot candidates."""
import math


def choose_candidates(choices, dense_ppl, budget, *, require_budget=True):
    """Each slot has ordered candidates containing compression and perplexity.

    Climb one rung at a time. Ties favor larger byte gain, then earlier layers
    and keys. Fixed-rung callers may allow a best effort when a budget is unreachable.
    """
    if not choices or not math.isfinite(budget) or budget < 0:
        raise ValueError("need nonempty slots and a finite nonnegative budget")
    slots = list(choices)
    indices = {slot: -1 for slot in slots}
    total = 0.
    while total / len(slots) < budget - 1e-12:
        options = []
        for slot in slots:
            current_idx = indices[slot]
            next_idx = current_idx + 1
            if next_idx >= len(choices[slot]):
                continue
            current = None if current_idx == -1 else choices[slot][current_idx]
            nxt = choices[slot][next_idx]
            gain = nxt["compression"] - (current["compression"] if current else 0.)
            if gain <= 0:
                raise ValueError("calibrated rung compression must increase")
            cost = nxt["ppl"] - (current["ppl"] if current else dense_ppl)
            efficiency = float("inf") if cost <= 0 else gain / cost
            options.append((efficiency, gain, -slot[0], slot[1] == "k", slot, next_idx))
        if not options:
            if require_budget:
                raise ValueError(f"budget {budget} unreachable")
            break
        _, gain, _, _, slot, next_idx = max(options)
        indices[slot] = next_idx
        total += gain
    return {slot: choices[slot][index] for slot, index in indices.items() if index >= 0}, total


def allocate(calibration, layers, targets, budget):
    slots = [(layer, target) for layer in range(layers) for target in targets]
    choices = {slot: sorted([r for r in calibration["selected"]
                            if (r["layer"], r["target"]) == slot], key=lambda r: r["budget"]) for slot in slots}
    selected, total = choose_candidates(choices, calibration["dense_ppl"], budget)
    assignment = list(selected.values())
    return {"budget": budget, "compression": total / len(slots),
            "whole_kv_compression": total / (2 * layers), "assignment": assignment,
            "kv": {"pipeline": [step for r in assignment for step in r["kv"]["pipeline"]]}}
