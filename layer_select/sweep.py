"""Build multi_run configurations: dense plus each slot at each compression level."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from catalog.compressions import DENSE
from layer_select.apply import kv_for_slot
from layer_select.levels import DEFAULT_LEVELS
from layer_select.slots import Slot, space_slots


def expand_singleton_configs(
    *,
    n_layers: int,
    space: str,
    method_kv: dict,
    method_tag: str,
    results_dir: str,
    name_prefix: str,
    extra: dict[str, Any] | None = None,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
) -> list[dict]:
    """Dense first, then one configuration per (slot, level)."""
    extra = extra or {}
    configurations = [
        _make_configuration(
            name=f"{name_prefix}_dense",
            kv=DENSE,
            results_dir=results_dir,
            extra=extra,
        )
    ]
    for slot in space_slots(space, n_layers):
        for pct in levels:
            configurations.append(
                _make_configuration(
                    name=f"{name_prefix}_{method_tag}_{slot.tag()}_p{pct}",
                    kv=kv_for_slot(method_kv, slot, level=pct),
                    results_dir=results_dir,
                    extra=extra,
                    slot=slot,
                    level=pct,
                )
            )
    return configurations


def _make_configuration(name, kv, results_dir, extra, slot: Slot | None = None, level: int | None = None):
    configuration = {
        "name": name,
        "kv": deepcopy(kv),
        "output_path": f"{results_dir.rstrip('/')}/{name}.json",
    }
    configuration.update(deepcopy(extra))
    if slot is not None:
        configuration["layer_slot"] = slot.to_dict()
    if level is not None:
        configuration["compression_level"] = level
    return configuration
