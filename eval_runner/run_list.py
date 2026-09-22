"""Load a Python or JSON run list into defaults plus a list of runs."""

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


def load_run_list(path: Path) -> dict:
    """Load a Python or JSON run list into ``{...defaults, "runs": [...]}``.

    A ``.py`` file must define ``config()`` returning that dict, or a list of runs.
    JSON is still accepted for a one-off hand-written list.
    """
    path = path.resolve()
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("_run_list", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import run list {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "config", None)):
            raise ValueError(f"{path} must define config()")
        loaded = module.config()
    else:
        loaded = json.loads(path.read_text())
    if isinstance(loaded, list):
        return {"runs": loaded}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must return a dict or a list of runs")
    return loaded


def split_base_and_runs(config: dict) -> tuple[dict, list]:
    base = {
        key: value
        for key, value in config.items()
        if key != "runs" and not str(key).startswith("_")
    }
    return base, config.get("runs") or []


def merge(base, run, key, default=None):
    if key in run:
        return run[key]
    if key in base:
        return base[key]
    return default


def reject_deprecated_kv_keys(base, run):
    found = [key for key in _DEPRECATED_KV_KEYS if key in base or key in run]
    if found:
        raise ValueError(
            "Deprecated compression keys are no longer read: "
            f"{found}. Put methods under kv.pipeline instead."
        )
