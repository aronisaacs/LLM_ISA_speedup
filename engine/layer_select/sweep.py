"""Build multi_run configurations: dense plus each key and value at 25/50/75%."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from catalog.compressions import DENSE
from engine.layer_select.apply import kv_for_slot
from engine.layer_select.rungs import rungs_for
from engine.layer_select.slots import Slot, all_slots


def expand_singleton_configs(
    *,
    n_layers: int,
    method_kv: dict,
    method_tag: str,
    results_dir: str,
    name_prefix: str,
    extra: dict[str, Any] | None = None,
) -> list[dict]:
    """Dense first, then one configuration per (slot, rung).

    Rungs come from the method: prune-style methods use 25/50/75, and an
    on/off method uses its single stored-ratio rung.
    """
    extra = extra or {}
    rungs = rungs_for(method_kv["pipeline"][0]["method"])
    configurations = [
        _make_configuration(
            name=f"{name_prefix}_dense",
            kv=DENSE,
            results_dir=results_dir,
            extra=extra,
        )
    ]
    for slot in all_slots(n_layers):
        for rung in rungs:
            pct = rung.level
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
