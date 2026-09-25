"""Reuse a finished lm-eval JSON when the simulation was already scored.

The key is the model, the task settings, and the kv pipeline. The output path
is not part of it, so a later study can copy an earlier JSON instead of rerunning.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from engine.eval_runner.execute import is_finished_result, normalize_tasks
from engine.eval_runner.load_run import merge
from engine.kv_compress.spec import parse_kv_spec

_INDEX: dict[str, dict[str, Path]] = {}
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


def reuse_cached_result(base, configuration, kv_spec, output_path: Path, *, skip_unkeyed: bool) -> str | None:
    """Copy a matching JSON onto ``output_path`` when one already exists.

    Returns ``"skip"`` when the destination is already that simulation,
    ``"reused"`` when another file was copied, or ``None`` when nothing matches.
    """
    identity = simulation_identity(base, configuration, kv_spec)
    if is_finished_result(output_path) and _same_simulation(output_path, identity):
        return "skip"
    if is_finished_result(output_path) and skip_unkeyed and _identity_from_file(output_path) is None:
        return "skip"
    cached = _find(identity, output_path)
    if cached is None:
        return None
    if cached.resolve() == output_path.resolve():
        return "skip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, output_path)
    _remember(output_path, identity)
    return "reused"


def stamp_simulation(results: dict, identity: dict) -> dict:
    stamped = dict(results)
    stamped["simulation"] = identity
    return stamped


def _same_simulation(path: Path, identity: dict) -> bool:
    stored = _identity_from_file(path)
    if stored is None:
        return False
    if _canonical(stored) == _canonical(identity):
        return True
    return _canonical(_legacy(stored)) == _canonical(_legacy(identity))


def _find(identity: dict, output_path: Path) -> Path | None:
    wanted = _canonical(identity)
    legacy = _canonical(_legacy(identity))
    for root in _roots(output_path):
        index = _index(root)
        hit = index.get(wanted) or index.get(legacy)
        if hit is not None and is_finished_result(hit):
            return hit
    return None


def _roots(output_path: Path) -> list[Path]:
    roots = []
    repo = Path(__file__).resolve().parents[2]
    repo_results = repo / "results"
    if repo_results.is_dir():
        roots.append(repo_results)
    nearest = output_path.parent
    while not nearest.exists():
        if nearest == nearest.parent:
            return roots
        nearest = nearest.parent
    if nearest.resolve() not in {root.resolve() for root in roots} and nearest != repo:
        roots.append(nearest)
    return roots


def _index(root: Path) -> dict[str, Path]:
    key = str(root.resolve())
    cached = _INDEX.get(key)
    if cached is not None:
        return cached
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        identity = _identity_from_file(path)
        if identity is None:
            continue
        index.setdefault(_canonical(identity), path)
        index.setdefault(_canonical(_legacy(identity)), path)
    _INDEX[key] = index
    return index


def _remember(path: Path, identity: dict) -> None:
    for root, index in _INDEX.items():
        if str(path.resolve()).startswith(root):
            index[_canonical(identity)] = path
            index[_canonical(_legacy(identity))] = path


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
