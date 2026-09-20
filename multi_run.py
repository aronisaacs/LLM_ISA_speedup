#!/usr/bin/env python3

import argparse
import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

from lm_eval.api.registry import get_model
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import simple_parse_args_string

from kv_compress import install, parse_kv_spec

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


def merge(base, run, key, default=None):
    if key in run:
        return run[key]
    if key in base:
        return base[key]
    return default


def normalize_tasks(tasks):
    if tasks is None:
        return []
    if isinstance(tasks, str):
        return [task.strip() for task in tasks.split(",") if task.strip()]
    return [copy.deepcopy(task) if isinstance(task, dict) else str(task) for task in tasks]


def normalize_samples(samples):
    if samples is None:
        return None
    if isinstance(samples, dict):
        return samples
    if isinstance(samples, str):
        text = samples.strip()
        if not text:
            return None
        if text.startswith("@"):
            text = text[1:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(Path(text).read_text())
    return samples


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


def reject_deprecated_kv_keys(base, run):
    found = [key for key in _DEPRECATED_KV_KEYS if key in base or key in run]
    if found:
        raise ValueError(
            "Deprecated compression keys are no longer read: "
            f"{found}. Put methods under kv.pipeline instead."
        )


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


def json_default(obj):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if isinstance(config, list):
        config = {"runs": config}

    base = {key: value for key, value in config.items() if key != "runs"}
    runs = config.get("runs") or []

    default_tasks = normalize_tasks(base.get("tasks"))
    default_samples = normalize_samples(base.get("samples"))
    loaded_model_key = None
    lm = None

    for run in runs:
        run_name = run.get("name", "run")
        reject_deprecated_kv_keys(base, run)
        model_name = merge(base, run, "model", "hf")
        model_args = merge(base, run, "model_args", "")
        serialized_model_args = (
            json.dumps(model_args, sort_keys=True)
            if isinstance(model_args, dict)
            else model_args
        )
        model_key = (model_name, serialized_model_args)
        if model_key != loaded_model_key:
            model_cls = get_model(model_name)
            lm = (
                model_cls.create_from_arg_obj(model_args)
                if isinstance(model_args, dict)
                else model_cls.create_from_arg_string(model_args)
            )
            loaded_model_key = model_key

        base_model_args = (
            dict(model_args)
            if isinstance(model_args, dict)
            else simple_parse_args_string(model_args)
        )
        tasks = normalize_tasks(merge(base, run, "tasks", default_tasks))
        samples = normalize_samples(merge(base, run, "samples", default_samples))
        env = normalize_env(merge(base, run, "env", None))
        kv_spec = parse_kv_spec(merge(base, run, "kv", None))

        batch_size = normalize_batch_size(merge(base, run, "batch_size", None))
        max_batch_size = merge(base, run, "max_batch_size", None)
        apply_batch_size(lm, batch_size, max_batch_size)

        model_args_out = dict(base_model_args)
        model_args_out["kv"] = kv_spec.to_dict()

        metadata = merge(base, run, "metadata", None)
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
        metadata["kv"] = kv_spec.to_dict()

        uninstall = install(lm, kv_spec)
        try:
            with temporary_environ(env):
                results = simple_evaluate(
                    model=lm,
                    model_args=model_args_out,
                    tasks=tasks,
                    num_fewshot=merge(base, run, "num_fewshot", None),
                    batch_size=batch_size,
                    max_batch_size=max_batch_size,
                    device=merge(base, run, "device", None),
                    limit=merge(base, run, "limit", None),
                    samples=samples,
                    bootstrap_iters=merge(base, run, "bootstrap_iters", 100000),
                    write_out=merge(base, run, "write_out", False),
                    log_samples=merge(base, run, "log_samples", False),
                    system_instruction=merge(base, run, "system_instruction", None),
                    apply_chat_template=merge(base, run, "apply_chat_template", False),
                    fewshot_as_multiturn=merge(base, run, "fewshot_as_multiturn", True),
                    gen_kwargs=merge(base, run, "gen_kwargs", None),
                    predict_only=merge(base, run, "predict_only", False),
                    random_seed=merge(base, run, "random_seed", 0),
                    numpy_random_seed=merge(base, run, "numpy_random_seed", 1234),
                    torch_random_seed=merge(base, run, "torch_random_seed", 1234),
                    fewshot_random_seed=merge(base, run, "fewshot_random_seed", 1234),
                    confirm_run_unsafe_code=merge(base, run, "confirm_run_unsafe_code", False),
                    metadata=metadata,
                )
        finally:
            uninstall()

        if getattr(lm, "rank", 0) == 0:
            output_path = Path(merge(base, run, "output_path", f"{run_name}.json"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, default=json_default))
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
