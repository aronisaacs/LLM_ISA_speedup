"""Turn a method kv dict into singleton or mixed-level pipelines."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from catalog.compressions import DENSE
from engine.layer_select.slots import Slot


def method_template(kv: dict) -> dict:
    """Copy method kwargs with empty layer lists."""
    payload = deepcopy(kv)
    steps = payload.get("pipeline") or []
    if len(steps) != 1:
        raise ValueError("method kv must have exactly one pipeline step")
    steps[0]["k_layers"] = []
    steps[0]["v_layers"] = []
    return payload


def kv_for_slot(method_kv: dict, slot: Slot, level: int | None = None) -> dict:
    """One pipeline step with only this slot enabled, optionally at ``level`` %."""
    payload = deepcopy(method_kv)
    steps = payload.get("pipeline") or []
    if len(steps) != 1:
        raise ValueError("sweep method kv must have exactly one pipeline step")
    step = steps[0]
    if slot.target == "k":
        step["k_layers"] = [slot.layer]
        step["v_layers"] = []
    else:
        step["k_layers"] = []
        step["v_layers"] = [slot.layer]
    if level is not None:
        _set_level(step, level)
    return payload


def kv_for_assignment(method_kv: dict, assignment: dict[Slot, int]) -> dict:
    """One pipeline step per compression level (a slot cannot appear twice)."""
    by_level: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"k": [], "v": []})
    for slot, pct in assignment.items():
        if pct <= 0:
            continue
        by_level[int(pct)][slot.target].append(slot.layer)
    if not by_level:
        return deepcopy(DENSE)
    template = method_template(method_kv)
    base_step = template["pipeline"][0]
    steps = []
    for pct in sorted(by_level):
        step = deepcopy(base_step)
        step["k_layers"] = sorted(set(by_level[pct]["k"]))
        step["v_layers"] = sorted(set(by_level[pct]["v"]))
        _set_level(step, pct)
        if step["k_layers"] or step["v_layers"]:
            steps.append(step)
    return {"pipeline": steps} if steps else deepcopy(DENSE)


def kv_for_slots(method_kv: dict, slots, level: int | None = None) -> dict:
    """Uniform level on every chosen slot (legacy on/off apply)."""
    if level is None:
        level = _level_from_step((method_kv.get("pipeline") or [{}])[0]) or 50
    assignment = {slot: int(level) for slot in slots}
    return kv_for_assignment(method_kv, assignment)


def parse_singleton(kv: dict | None) -> tuple[Slot | None, int | None]:
    """Dense → (None, None). Singleton slot at one level → (Slot, pct)."""
    pipeline = (kv or {}).get("pipeline") or []
    if not pipeline:
        return None, None
    if len(pipeline) != 1:
        raise ValueError("expected dense or a single pipeline step")
    step = pipeline[0]
    k_layers = _as_index_list(step.get("k_layers"), "k_layers")
    v_layers = _as_index_list(step.get("v_layers"), "v_layers")
    if len(k_layers) == 1 and not v_layers:
        slot = Slot(k_layers[0], "k")
    elif len(v_layers) == 1 and not k_layers:
        slot = Slot(v_layers[0], "v")
    else:
        raise ValueError("kv is not a singleton slot")
    level = _level_from_step(step)
    if level is None:
        from engine.layer_select.rungs import RUNGS

        method_rungs = RUNGS.get(step.get("method"))
        if method_rungs is not None and len(method_rungs) == 1:
            level = method_rungs[0].level
    return slot, level


def slot_from_kv(kv: dict | None) -> Slot | None:
    slot, _level = parse_singleton(kv)
    return slot


def _set_level(step: dict, pct: int) -> None:
    method = step.get("method")
    if method == "sparsify_nm":
        n = int(step.get("n") or 8)
        keep = int(round(n * (1 - pct / 100.0)))
        step["m"] = min(max(keep, 0), n)
        return
    if method in {"vector_compress", "checksparse_l1"}:
        step["prune_pct"] = int(pct)
        if method == "vector_compress":
            step.pop("threshold", None)


def _level_from_step(step: dict) -> int | None:
    if not step:
        return None
    if step.get("prune_pct") is not None:
        return int(step["prune_pct"])
    if step.get("method") == "sparsify_nm":
        n = int(step["n"])
        m = int(step["m"])
        return int(round(100 * (n - m) / n))
    return None


def _as_index_list(value, label: str) -> list[int]:
    if value is None or value == []:
        return []
    if value == "all":
        raise ValueError(f"{label}=all is not a singleton")
    if isinstance(value, bool) or not isinstance(value, list):
        raise TypeError(f"{label} must be a list of indices")
    return [int(item) for item in value]
