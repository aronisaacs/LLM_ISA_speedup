"""Load a run file into shared defaults plus a list of configurations.

A run is a sequence of configurations (one model + compression + task each).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_DEPRECATED_KV_KEYS = (
    "sparsity_4_to_8_enabled",
    "sparsity_4_to_8_k_layers",
    "sparsity_4_to_8_v_layers",
    "tile_sparsity_enabled",
    "tile_sparsity_tile",
    "tile_sparsity_prune_pct_by_target_layer",
    "kv_quantization_enabled",
    "kv_quantization_group_size",
    "kv_quantization_bits_by_target_layer",
)


def load_run(path: Path) -> dict:
    """Load a Python or JSON run into ``{...defaults, "configurations": [...]}``.

    A ``.py`` file must define ``run()`` returning that dict, or a list of
    configurations. JSON is still accepted for a one-off list.
    """
    path = path.resolve()
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("_run_file", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import run {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run", None)):
            raise ValueError(f"{path} must define run()")
        loaded = module.run()
    else:
        loaded = json.loads(path.read_text())
    if isinstance(loaded, list):
        return {"configurations": loaded}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must return a dict or a list of configurations")
    return loaded


def split_base_and_configurations(run: dict) -> tuple[dict, list]:
    base = {
        key: value
        for key, value in run.items()
        if key != "configurations" and not str(key).startswith("_")
    }
    return base, run.get("configurations") or []


def merge(base, configuration, key, default=None):
    if key in configuration:
        return configuration[key]
    if key in base:
        return base[key]
    return default


def reject_deprecated_kv_keys(base, configuration):
    found = [key for key in _DEPRECATED_KV_KEYS if key in base or key in configuration]
    if found:
        raise ValueError(
            "Deprecated compression keys are no longer read: "
            f"{found}. Put methods under kv.pipeline instead."
        )
