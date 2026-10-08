"""Process-local compression observations for one evaluation.

Kernels report representation bytes in STATS. The cache records dense bytes for
all slots, including unchanged slots, so totals do not assume equal layer sizes.
Byte counts describe simulated storage, not allocated PyTorch memory.
"""
STATS = {}
_OBSERVED = {}
_SELECTED = set()
_OVERLAPPING = set()


def reset_stats():
    STATS.clear()
    _OBSERVED.clear()
    _SELECTED.clear()
    _OVERLAPPING.clear()


def pop_stats():
    snapshot = {key: dict(value) for key, value in STATS.items()}
    reset_stats()
    return snapshot


def observe(keys, values, layer, spec):
    for target, tensor in (("k", keys), ("v", values)):
        key = f"{target}_layer_{layer}"
        # Current simulation formats define dense vectors as 16-bit values.
        _OBSERVED[key] = _OBSERVED.get(key, 0) + tensor.numel() * 16
        steps = [step for step in spec.pipeline
                 if getattr(step, target + "_layers") == "all" or
                 layer in getattr(step, target + "_layers")]
        if len(steps) > 1:
            _OVERLAPPING.add(key)
        elif steps:
            _SELECTED.add(key)


def storage_summary():
    """Return measured savings only when every selected slot has byte accounting."""
    if not _OBSERVED:
        return None
    if _OVERLAPPING:
        return None  # Composition needs an explicit combined representation.
    for key in _SELECTED:
        row = STATS.get(key, {})
        if "dense_bits" not in row or "stored_bits" not in row:
            return None
    totals = {}
    for target in ("k", "v"):
        dense = sum(value for key, value in _OBSERVED.items() if key.startswith(target + "_"))
        saved = sum(row["dense_bits"] - row["stored_bits"] for key, row in STATS.items()
                    if key.startswith(target + "_layer_") and "dense_bits" in row)
        totals[target] = {"dense_bits": dense, "stored_bits": dense - saved,
                          "compression": saved / dense if dense else 0.}
    dense = sum(row["dense_bits"] for row in totals.values())
    stored = sum(row["stored_bits"] for row in totals.values())
    totals["kv"] = {"dense_bits": dense, "stored_bits": stored,
                    "compression": 1 - stored / dense if dense else 0.}
    return {"scope": "simulated_cache_storage", "dense_value_bits": 16, "targets": totals}
