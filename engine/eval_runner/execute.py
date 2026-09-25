"""lm-eval load / evaluate / write helpers used by multi_run.py."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

from engine.eval_runner.device import apply_device, available_device
from engine.eval_runner.load_run import merge

_EVAL_DEFAULTS = {
    "num_fewshot": None,
    "max_batch_size": None,
    "limit": None,
    "bootstrap_iters": 100000,
    "write_out": False,
    "log_samples": False,
    "system_instruction": None,
    "apply_chat_template": False,
    "fewshot_as_multiturn": True,
    "gen_kwargs": None,
    "predict_only": False,
    "random_seed": 0,
    "numpy_random_seed": 1234,
    "torch_random_seed": 1234,
    "fewshot_random_seed": 1234,
    "confirm_run_unsafe_code": False,
}


def load_model_if_needed(lm, loaded_model_key, base, configuration):
    """Reuse the HF model when model_args match the previous configuration."""
    from lm_eval.api.registry import get_model

    model_name = merge(base, configuration, "model", "hf")
    device = merge(base, configuration, "device", None) or available_device()
    model_args = apply_device(merge(base, configuration, "model_args", ""), device)
    serialized = (
        json.dumps(model_args, sort_keys=True) if isinstance(model_args, dict) else model_args
    )
    model_key = (model_name, serialized)
    if model_key != loaded_model_key:
        model_cls = get_model(model_name)
        lm = (
            model_cls.create_from_arg_obj(model_args)
            if isinstance(model_args, dict)
            else model_cls.create_from_arg_string(model_args)
        )
        loaded_model_key = model_key
    return lm, loaded_model_key, device, model_args


def evaluate(lm, base, configuration, kv_spec, device, model_args):
    """One lm-eval simple_evaluate call with KV metadata attached."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.utils import simple_parse_args_string

    tasks = normalize_tasks(merge(base, configuration, "tasks", None))
    env = normalize_env(merge(base, configuration, "env", None))
    batch_size = normalize_batch_size(merge(base, configuration, "batch_size", None))
    max_batch_size = merge(base, configuration, "max_batch_size", None)
    apply_batch_size(lm, batch_size, max_batch_size)

    parsed_args = dict(model_args) if isinstance(model_args, dict) else simple_parse_args_string(model_args)
    parsed_args["kv"] = kv_spec.to_dict()
    metadata = dict(merge(base, configuration, "metadata", None) or {})
    metadata["kv"] = kv_spec.to_dict()

    eval_kwargs = {
        key: merge(base, configuration, key, default) for key, default in _EVAL_DEFAULTS.items()
    }
    with temporary_environ(env):
        return simple_evaluate(
            model=lm,
            model_args=parsed_args,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            metadata=metadata,
            verbosity="INFO",
            **eval_kwargs,
        )


def result_output_path(base, configuration) -> Path:
    name = configuration.get("name", "configuration")
    return Path(merge(base, configuration, "output_path", f"{name}.json"))


def is_finished_result(path: Path) -> bool:
    """True when ``path`` is an lm-eval JSON that already has task scores."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    results = payload.get("results")
    return isinstance(results, dict) and bool(results)


def shorten_result(payload: dict) -> dict:
    """Keep the simulation and its scores. Drop repeated task templates and the env dump."""
    if not isinstance(payload, dict):
        return payload
    tasks = _score_tasks(payload)
    short = {"results": {task: _metrics(payload, task) for task in tasks}}
    short["results"] = {task: metrics for task, metrics in short["results"].items() if metrics}
    config = _short_config(payload)
    if config:
        short["config"] = config
    shots = _one_shot(payload, tasks)
    if shots:
        short["n-shot"] = shots
    samples = _one_sample_count(payload, tasks)
    if samples:
        short["n-samples"] = samples
    budget = _budget_metadata(payload)
    if budget and tasks:
        short["configs"] = {tasks[0]: {"metadata": budget}}
    if isinstance(payload.get("simulation"), dict):
        short["simulation"] = payload["simulation"]
    else:
        from engine.eval_runner.cache import _legacy_from_payload

        identity = _legacy_from_payload(payload)
        if identity is not None:
            short["simulation"] = identity
    return short


def write_result_json(lm, base, configuration, results, simulation=None):
    if getattr(lm, "rank", 0) != 0:
        return None
    if simulation is not None:
        results = dict(results)
        results["simulation"] = simulation
    results = shorten_result(results)
    output_path = result_output_path(base, configuration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=_json_default))
    return output_path


def _score_tasks(payload: dict) -> list[str]:
    groups = payload.get("groups") or {}
    if groups:
        return list(groups)
    children: set[str] = set()
    for kids in (payload.get("group_subtasks") or {}).values():
        children.update(kids or [])
    results = payload.get("results") or {}
    parents = [task for task in results if task not in children]
    return parents or list(results)


def _metrics(payload: dict, task: str) -> dict:
    source = (payload.get("groups") or {}).get(task) or (payload.get("results") or {}).get(task) or {}
    if not isinstance(source, dict):
        return {}
    return {key: value for key, value in source.items() if key != "alias" and "stderr" not in key}


def _short_config(payload: dict) -> dict:
    config = payload.get("config") or {}
    model_args = config.get("model_args") if isinstance(config.get("model_args"), dict) else {}
    kept = {key: model_args[key] for key in ("pretrained", "dtype") if key in model_args}
    try:
        from engine.eval_runner.cache import _kv_from_payload

        kept["kv"] = _kv_from_payload(payload)
    except (TypeError, ValueError):
        if "kv" in model_args:
            kept["kv"] = model_args["kv"]
    short = {"model_args": kept} if kept else {}
    for key in ("limit", "gen_kwargs"):
        if config.get(key) is not None:
            short[key] = config[key]
    return short


def _one_shot(payload: dict, tasks: list[str]) -> dict:
    shots = payload.get("n-shot") or {}
    values = list(shots.values())
    if not values or not tasks or any(value != values[0] for value in values):
        return {}
    return {tasks[0]: values[0]}


def _one_sample_count(payload: dict, tasks: list[str]) -> dict:
    raw = payload.get("n-samples") or {}
    if not tasks:
        return {}
    info = raw.get(tasks[0])
    if isinstance(info, dict):
        return {tasks[0]: {key: info[key] for key in ("original", "effective") if key in info}}
    original = effective = 0
    saw = False
    for info in raw.values():
        if not isinstance(info, dict):
            continue
        saw = True
        original += info.get("original") or 0
        effective += info.get("effective") or 0
    if not saw:
        return {}
    return {tasks[0]: {"original": original, "effective": effective}}


def _budget_metadata(payload: dict) -> dict:
    for task_config in (payload.get("configs") or {}).values():
        metadata = (task_config or {}).get("metadata") or {}
        if "kv_budget" in metadata and "kv_compression" in metadata:
            return {"kv_budget": metadata["kv_budget"], "kv_compression": metadata["kv_compression"]}
    return {}


def normalize_tasks(tasks):
    if tasks is None:
        return []
    if isinstance(tasks, str):
        return [task.strip() for task in tasks.split(",") if task.strip()]
    return [copy.deepcopy(task) if isinstance(task, dict) else str(task) for task in tasks]


def normalize_env(env):
    if env is None:
        return {}
    return {str(key): str(value) for key, value in env.items()}


def normalize_batch_size(batch_size):
    if batch_size is None:
        return batch_size
    if isinstance(batch_size, int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return batch_size
    text = str(batch_size).strip().lower()
    if text == "auto":
        return text
    if text.startswith("auto:"):
        schedule = text.removeprefix("auto:")
        if not schedule.isdigit() or int(schedule) < 1:
            raise ValueError("batch_size auto schedule must be a positive integer, e.g. 'auto:4'")
        return text
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError("batch_size must be a positive integer, 'auto', or 'auto:N'") from error
    if value < 1:
        raise ValueError("batch_size must be positive")
    return value


def apply_batch_size(lm, batch_size, max_batch_size):
    """Update a reused lm-eval model with the same semantics as HFLM.__init__."""
    if batch_size is not None:
        if isinstance(batch_size, str) and batch_size.startswith("auto"):
            parts = batch_size.split(":", 1)
            lm.batch_size_per_gpu = "auto"
            lm.batch_schedule = float(parts[1]) if len(parts) == 2 else 1.0
        else:
            lm.batch_size_per_gpu = int(batch_size)
            lm.batch_schedule = 1.0
        if hasattr(lm, "batch_sizes"):
            lm.batch_sizes.clear()
    if max_batch_size is not None:
        lm.max_batch_size = int(max_batch_size)


@contextmanager
def temporary_environ(updates):
    if not updates:
        yield
        return

    previous = {}
    missing = []
    for key, value in updates.items():
        if key in os.environ:
            previous[key] = os.environ[key]
        else:
            missing.append(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, value in previous.items():
            os.environ[key] = value
        for key in missing:
            os.environ.pop(key, None)


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)
