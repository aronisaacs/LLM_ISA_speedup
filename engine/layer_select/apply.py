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
    by_level: dict[int | float, dict[str, list[int]]] = defaultdict(lambda: {"k": [], "v": []})
    for slot, pct in assignment.items():
        if pct <= 0:
            continue
        by_level[_level(pct)][slot.target].append(slot.layer)
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
    assignment = {slot: _level(level) for slot in slots}
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


# TurboQuant's half-bit rung: bits + 0.5 = half of a 128-channel head at bits + 1.
TURBOQUANT_HALF_OUTLIERS = 64
TURBOQUANT_HEAD_DIM = 128


def _level(pct):
    """Whole levels stay ints (keys and names as before); TurboQuant's 2.5 stays 2.5."""
    return int(pct) if float(pct).is_integer() else float(pct)


def _set_level(step: dict, pct) -> None:
    method = step.get("method")
    if method == "turboquant_mse":
        _set_turboquant_level(step, pct)
        return
    if method == "quantize":
        step["bits"] = int(pct)
        return
    if method in {"vector_compress", "vector_compress_pair",
                  "checksparse_l1", "checksparse_row"}:
        step["prune_pct"] = int(pct)
        if method == "vector_compress":
            step.pop("threshold", None)


def _level_from_step(step: dict) -> int | None:
    if not step:
        return None
    if step.get("method") == "turboquant_mse" and step.get("bits") is not None:
        bits, outliers = int(step["bits"]), int(step.get("outliers") or 0)
        high = int(step.get("outlier_bits") or bits + 1)
        return _level(bits + outliers * (high - bits) / TURBOQUANT_HEAD_DIM)
    if step.get("method") == "quantize" and step.get("bits") is not None:
        return int(step["bits"])
    if step.get("prune_pct") is not None:
        return int(step["prune_pct"])
    return None


def _set_turboquant_level(step: dict, level) -> None:
    """Whole bits: plain TurboQuant. bits + 0.5: 64 calibrated outlier channels at bits + 1."""
    from catalog.compressions import TURBOQUANT_OUTLIER_TABLE

    bits, half = divmod(float(level), 1)
    if half not in (0.0, 0.5):
        raise ValueError(f"turboquant_mse level must be whole or half bits, got {level}")
    step["bits"] = int(bits)
    step.pop("outlier_bits", None)
    if half:
        step["outliers"] = TURBOQUANT_HALF_OUTLIERS
        step.setdefault("outlier_table", TURBOQUANT_OUTLIER_TABLE)
    else:
        step.pop("outliers", None)
        step.pop("outlier_table", None)


def _as_index_list(value, label: str) -> list[int]:
    if value is None or value == []:
        return []
    if value == "all":
        raise ValueError(f"{label}=all is not a singleton")
    if isinstance(value, bool) or not isinstance(value, list):
        raise TypeError(f"{label} must be a list of indices")
    return [int(item) for item in value]
