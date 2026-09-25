"""Load a run file into shared defaults plus a list of configurations.

A run is a sequence of configurations (one model + compression + task each).
"""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
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


def load_run(path: Path | str) -> dict:
    """Load a Python or JSON run into ``{...defaults, "configurations": [...]}``.

    A ``.py`` file must define ``run()``, or ``path.py:function`` selects another
    function in that file. Either returns that dict, or a list of configurations.
    JSON is still accepted for a one-off list.
    """
    path, function = _split_selector(path)
    path = path.resolve()
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("_run_file", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import run {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = function or "run"
        chosen = getattr(module, name, None)
        if not callable(chosen):
            raise ValueError(f"{path} must define {name}()")
        loaded = chosen()
    else:
        if function is not None:
            raise ValueError(f"JSON runs have no function selector, got {function}")
        loaded = json.loads(path.read_text())
    if isinstance(loaded, list):
        return {"configurations": loaded}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must return a dict or a list of configurations")
    return loaded


def grid(*, results: str, methods, tasks, model_tag: str = "llama31", model_args: str | None = None) -> list[dict]:
    """One configuration per method, then per task.

    ``methods`` is ``(tag, kv)`` pairs. ``tasks`` are catalog task dicts.
    ``name_task`` and ``file`` name the configuration and are not copied onto it.
    A single task with no ``name_task`` leaves the task out of the name.
    ``file`` selects ``{results}/{file}_{method}.json``; otherwise the path is the name.
    """
    several_tasks = len(tuple(tasks)) > 1
    configurations = []
    for method_tag, kv in methods:
        for task in tasks:
            task_tag = task.get("name_task")
            if task_tag is None and several_tasks:
                task_tag = task["tasks"][0]
            name = f"{model_tag}_{method_tag}" if task_tag is None else f"{model_tag}_{task_tag}_{method_tag}"
            if "file" in task:
                output_path = f"{results}/{task['file']}_{method_tag}.json"
            else:
                output_path = f"{results}/{name}.json"
            configuration = {
                "name": name,
                "kv": deepcopy(kv),
                "output_path": output_path,
            }
            if model_args is not None:
                configuration["model_args"] = model_args
            extra = {key: deepcopy(value) for key, value in task.items() if key not in {"name_task", "file"}}
            configuration.update(extra)
            configurations.append(configuration)
    return configurations


def _split_selector(path: Path | str) -> tuple[Path, str | None]:
    text = str(path)
    marker = ".py:"
    index = text.rfind(marker)
    if index == -1:
        return Path(text), None
    function = text[index + len(marker) :]
    if not function:
        raise ValueError(f"missing function after {text[: index + 3]}")
    return Path(text[: index + 3]), function


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
