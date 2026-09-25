"""Reuse a finished lm-eval JSON when the simulation was already scored.

The key is the model, the task settings, and the kv pipeline. The output path
is not part of it, so a later study can copy an earlier JSON instead of rerunning.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.eval_runner.execute import is_finished_result, normalize_tasks
from engine.eval_runner.load_run import merge
from engine.kv_compress.spec import parse_kv_spec

_LEGACY = (
    "pretrained",
    "dtype",
    "tasks",
    "num_fewshot",
    "limit",
    "gen_kwargs",
    "kv",
)


def simulation_identity(base, configuration, kv_spec) -> dict:
    """Fields that change the score. Device and batch size are left out."""
    model_args = merge(base, configuration, "model_args", "")
    fields = _model_fields(model_args)
    fields.update(
        {
            "tasks": sorted(normalize_tasks(merge(base, configuration, "tasks", None))),
            "num_fewshot": merge(base, configuration, "num_fewshot", None),
            "limit": merge(base, configuration, "limit", None),
            "gen_kwargs": merge(base, configuration, "gen_kwargs", None),
            "apply_chat_template": bool(merge(base, configuration, "apply_chat_template", False)),
            "kv": kv_spec.to_dict(),
        }
    )
    return fields


def reuse_cached_result(base, configuration, kv_spec, output_path=None, *, skip_unkeyed: bool = False, root: Path | None = None) -> str | None:
    """Return ``"skip"`` when this simulation is already in the index."""
    del output_path, skip_unkeyed
    from engine.eval_runner.index import find_result

    identity = simulation_identity(base, configuration, kv_spec)
    if find_result(identity, root) is None:
        return None
    return "skip"


def stamp_simulation(results: dict, identity: dict) -> dict:
    stamped = dict(results)
    stamped["simulation"] = identity
    return stamped


def _identity_from_file(path: Path) -> dict | None:
    if not is_finished_result(path):
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    stamped = payload.get("simulation")
    if isinstance(stamped, dict) and "kv" in stamped:
        return stamped
    return _legacy_from_payload(payload)


def _legacy_from_payload(payload: dict) -> dict | None:
    config = payload.get("config") or {}
    model_args = config.get("model_args") or {}
    if not isinstance(model_args, dict) or "pretrained" not in model_args:
        return None
    try:
        kv = parse_kv_spec(_kv_from_payload(payload)).to_dict()
    except (TypeError, ValueError):
        return None
    groups = payload.get("group_subtasks") or {}
    tasks = sorted(groups) if groups else sorted(payload.get("results") or {})
    shots = list((payload.get("n-shot") or {}).values())
    fewshot = shots[0] if shots and all(shot == shots[0] for shot in shots) else (shots or None)
    return {
        "pretrained": model_args.get("pretrained"),
        "dtype": model_args.get("dtype"),
        "tasks": tasks,
        "num_fewshot": fewshot,
        "limit": config.get("limit"),
        "gen_kwargs": config.get("gen_kwargs"),
        "kv": kv,
    }


def _kv_from_payload(payload: dict) -> dict:
    configs = payload.get("configs") or {}
    for task_config in configs.values():
        if not isinstance(task_config, dict):
            continue
        metadata = task_config.get("metadata") or {}
        if "kv" in metadata:
            return metadata["kv"] or {"pipeline": []}
    model_args = (payload.get("config") or {}).get("model_args") or {}
    if isinstance(model_args, dict) and "kv" in model_args:
        return model_args["kv"] or {"pipeline": []}
    raise ValueError("no kv metadata")


def _legacy(identity: dict) -> dict:
    return {key: identity.get(key) for key in _LEGACY}


def _model_fields(model_args) -> dict:
    if isinstance(model_args, dict):
        raw = model_args
    else:
        raw = {}
        for part in str(model_args).split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            raw[key.strip()] = value.strip()
    return {"pretrained": raw.get("pretrained"), "dtype": raw.get("dtype")}


def _canonical(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, default=str)
