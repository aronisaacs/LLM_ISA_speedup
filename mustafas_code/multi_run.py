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


KV_QUANTIZATION_BITS = (4, 8, 16)


def normalize_kv_quantization_bits_by_target_layer(value):
    """Validate and canonicalize the per-target KV quantization bit map."""
    if value is None:
        return {"k": {}, "v": {}}
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("@"):  # Explicit JSON path, matching existing runner options.
            text = text[1:]
        if text.startswith(("{", "[")):
            value = json.loads(text)
        else:
            value = json.loads(Path(text).read_text())
    if not isinstance(value, dict):
        raise TypeError("kv_quantization_bits_by_target_layer must be a mapping or JSON path")

    unknown_targets = set(value) - {"k", "v"}
    if unknown_targets:
        raise ValueError(
            "kv_quantization_bits_by_target_layer may contain only 'k' and 'v' targets; "
            f"got {sorted(unknown_targets, key=str)!r}"
        )

    normalized = {}
    for target in ("k", "v"):
        target_map = value.get(target, {})
        if target_map is None:
            raise TypeError(f"kv_quantization_bits_by_target_layer[{target!r}] must be a mapping")
        if not isinstance(target_map, dict):
            raise TypeError(f"kv_quantization_bits_by_target_layer[{target!r}] must be a mapping")

        normalized_target = {}
        for layer_idx, bits in target_map.items():
            if isinstance(layer_idx, bool):
                raise TypeError(f"{target} layer indices must be non-negative integers")
            if isinstance(layer_idx, int):
                normalized_layer_idx = layer_idx
            elif isinstance(layer_idx, str) and layer_idx.strip().isdigit():
                normalized_layer_idx = int(layer_idx.strip())
            else:
                raise TypeError(f"{target} layer indices must be non-negative integers")
            if normalized_layer_idx < 0:
                raise ValueError(f"{target} layer indices must be non-negative")
            if isinstance(bits, bool) or not isinstance(bits, int):
                raise TypeError(f"{target} layer bit-widths must be integers")
            if bits not in KV_QUANTIZATION_BITS:
                raise ValueError(f"{target} layer bit-widths must be one of {KV_QUANTIZATION_BITS}")
            normalized_target[normalized_layer_idx] = bits
        normalized[target] = normalized_target
    return normalized


def resolve_decoder_num_hidden_layers(lm):
    """Resolve the number of layers in the decoder used by a possibly composite model."""
    model = getattr(lm, "model", None)
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is not None:
        try:
            return len(layers)
        except TypeError:
            pass

    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    for config in (text_config, model_config, getattr(lm, "_config", None)):
        if isinstance(config, dict):
            num_hidden_layers = config.get("num_hidden_layers")
        else:
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int) and not isinstance(num_hidden_layers, bool):
            return num_hidden_layers
    return None


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


def normalize_tile_prune_map(value):
    if value is None:
        return {}
    if isinstance(value, str):
        path = value[1:] if value.startswith("@") else value
        value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("tile_sparsity_prune_pct_by_target_layer must be a mapping or JSON path")
    if "tile_sparsity_prune_pct_by_target_layer" in value:
        value = value["tile_sparsity_prune_pct_by_target_layer"]
    if not isinstance(value, dict):
        raise TypeError("allocation JSON must contain a prune_pct_by_target_layer mapping")
    return copy.deepcopy(value)


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


def apply_tile_sparsity_config(lm, values):
    model = lm.model
    model.config.tile_sparsity_enabled = bool(values.get("tile_sparsity_enabled", False))
    model.config.tile_sparsity_tile = int(values.get("tile_sparsity_tile", 8))
    model.config.tile_sparsity_prune_pct_by_target_layer = normalize_tile_prune_map(
        values.get("tile_sparsity_prune_pct_by_target_layer")
    )

    if hasattr(lm, "_config"):
        lm._config.tile_sparsity_enabled = model.config.tile_sparsity_enabled
        lm._config.tile_sparsity_tile = model.config.tile_sparsity_tile
        lm._config.tile_sparsity_prune_pct_by_target_layer = (
            model.config.tile_sparsity_prune_pct_by_target_layer
        )


def apply_4_to_8_sparsity_config(lm, values):
    configs = [lm.model.config]
    if hasattr(lm.model.config, "text_config"):
        configs.append(lm.model.config.text_config)
    if hasattr(lm, "_config") and lm._config not in configs:
        configs.append(lm._config)

    enabled = bool(values.get("sparsity_4_to_8_enabled", False))
    k_layers = list(values.get("sparsity_4_to_8_k_layers") or [])
    v_layers = list(values.get("sparsity_4_to_8_v_layers") or [])

    for model_config in configs:
        model_config.sparsity_4_to_8_enabled = enabled
        model_config.sparsity_4_to_8_k_layers = k_layers
        model_config.sparsity_4_to_8_v_layers = v_layers


def apply_kv_quantization_config(lm, values):
    """Apply validated adaptive KV QDQ settings to every relevant model config."""
    enabled = values.get("kv_quantization_enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("kv_quantization_enabled must be a boolean")

    group_size = values.get("kv_quantization_group_size", 64)
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError("kv_quantization_group_size must be a positive integer")
    if group_size <= 0:
        raise ValueError("kv_quantization_group_size must be a positive integer")

    bits_by_target_layer = normalize_kv_quantization_bits_by_target_layer(
        values.get("kv_quantization_bits_by_target_layer")
    )
    normalized = {
        "kv_quantization_enabled": enabled,
        "kv_quantization_group_size": group_size,
        "kv_quantization_bits_by_target_layer": bits_by_target_layer,
    }

    num_hidden_layers = resolve_decoder_num_hidden_layers(lm)
    selected_layers = [
        (target, layer_idx)
        for target, target_map in bits_by_target_layer.items()
        for layer_idx in target_map
    ]
    if selected_layers and num_hidden_layers is None:
        raise ValueError("Cannot validate KV quantization layers: decoder num_hidden_layers is unavailable")
    if num_hidden_layers is not None:
        for target, layer_idx in selected_layers:
            if layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"KV quantization {target} layer index {layer_idx} is out of range; "
                    f"decoder has {num_hidden_layers} layers (valid indices are 0..{num_hidden_layers - 1})"
                )

    configs = [lm.model.config]
    text_config = getattr(lm.model.config, "text_config", None)
    if text_config is not None and not any(text_config is config for config in configs):
        configs.append(text_config)
    lm_config = getattr(lm, "_config", None)
    if lm_config is not None and not any(lm_config is config for config in configs):
        configs.append(lm_config)

    for model_config in configs:
        model_config.kv_quantization_enabled = enabled
        model_config.kv_quantization_group_size = group_size
        model_config.kv_quantization_bits_by_target_layer = copy.deepcopy(bits_by_target_layer)
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if isinstance(config, list):
        config = {"runs": config}

    base = {
        key: value
        for key, value in config.items()
        if key != "runs"
    }
    runs = config.get("runs") or []

    default_tasks = normalize_tasks(base.get("tasks"))
    default_samples = normalize_samples(base.get("samples"))
    loaded_model_key = None
    lm = None

    for run in runs:
        run_name = run.get("name", "run")
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

        tile_values = {
            "tile_sparsity_enabled": merge(base, run, "tile_sparsity_enabled", False),
            "tile_sparsity_tile": merge(base, run, "tile_sparsity_tile", 8),
            "tile_sparsity_prune_pct_by_target_layer": normalize_tile_prune_map(
                merge(base, run, "tile_sparsity_prune_pct_by_target_layer", {})
            ),
        }
        apply_tile_sparsity_config(lm, tile_values)

        sparsity_4_to_8_values = {
            "sparsity_4_to_8_enabled": merge(base, run, "sparsity_4_to_8_enabled", False),
            "sparsity_4_to_8_k_layers": merge(base, run, "sparsity_4_to_8_k_layers", []),
            "sparsity_4_to_8_v_layers": merge(base, run, "sparsity_4_to_8_v_layers", []),
        }
        apply_4_to_8_sparsity_config(lm, sparsity_4_to_8_values)

        kv_quantization_values = {
            "kv_quantization_enabled": merge(base, run, "kv_quantization_enabled", False),
            "kv_quantization_group_size": merge(base, run, "kv_quantization_group_size", 64),
            "kv_quantization_bits_by_target_layer": merge(
                base, run, "kv_quantization_bits_by_target_layer", {}
            ),
        }
        kv_quantization_values = apply_kv_quantization_config(lm, kv_quantization_values)

        batch_size = normalize_batch_size(merge(base, run, "batch_size", None))
        max_batch_size = merge(base, run, "max_batch_size", None)
        apply_batch_size(lm, batch_size, max_batch_size)

        model_args_out = dict(base_model_args)
        model_args_out["tile_sparsity_enabled"] = bool(tile_values["tile_sparsity_enabled"])
        model_args_out["tile_sparsity_tile"] = int(tile_values["tile_sparsity_tile"])
        model_args_out["tile_sparsity_prune_pct_by_target_layer"] = tile_values[
            "tile_sparsity_prune_pct_by_target_layer"
        ]
        model_args_out["sparsity_4_to_8_enabled"] = bool(
            sparsity_4_to_8_values["sparsity_4_to_8_enabled"]
        )
        model_args_out["sparsity_4_to_8_k_layers"] = list(
            sparsity_4_to_8_values["sparsity_4_to_8_k_layers"] or []
        )
        model_args_out["sparsity_4_to_8_v_layers"] = list(
            sparsity_4_to_8_values["sparsity_4_to_8_v_layers"] or []
        )
        model_args_out["kv_quantization_enabled"] = kv_quantization_values["kv_quantization_enabled"]
        model_args_out["kv_quantization_group_size"] = kv_quantization_values["kv_quantization_group_size"]
        model_args_out["kv_quantization_bits_by_target_layer"] = kv_quantization_values[
            "kv_quantization_bits_by_target_layer"
        ]

        metadata = merge(base, run, "metadata", None)
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
        metadata["tile_sparsity"] = {
            "enabled": bool(tile_values["tile_sparsity_enabled"]),
            "tile": int(tile_values["tile_sparsity_tile"]),
            "prune_pct_by_target_layer": tile_values["tile_sparsity_prune_pct_by_target_layer"],
        }
        metadata["sparsity_4_to_8"] = {
            "enabled": bool(sparsity_4_to_8_values["sparsity_4_to_8_enabled"]),
            "k_layers": model_args_out["sparsity_4_to_8_k_layers"],
            "v_layers": model_args_out["sparsity_4_to_8_v_layers"],
        }
        metadata["kv_quantization"] = {
            "enabled": kv_quantization_values["kv_quantization_enabled"],
            "group_size": kv_quantization_values["kv_quantization_group_size"],
            "bits_by_target_layer": kv_quantization_values["kv_quantization_bits_by_target_layer"],
        }

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

        if getattr(lm, "rank", 0) == 0:
            output_path = Path(merge(base, run, "output_path", f"{run_name}.json"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, default=json_default))
            print(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
