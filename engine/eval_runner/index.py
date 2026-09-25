"""One row per finished simulation, independent of which run asked for it.

``results/index.json`` maps a model, task, and kv pipeline to the JSON that
holds the scores. A chart looks here instead of opening a run folder.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.eval_runner.cache import _canonical, _identity_from_file, _legacy
from engine.eval_runner.execute import is_finished_result

_NAME = "index.json"


def results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def index_path(root: Path | None = None) -> Path:
    return (root or results_root()) / _NAME


def find_result(identity: dict, root: Path | None = None) -> dict | None:
    """The index row for this simulation, or ``None`` when it has not been scored."""
    rows = _rows(root)
    wanted = {_canonical(identity), _canonical(_legacy(identity))}
    for row in rows:
        stored = row.get("identity") or {}
        if _canonical(stored) in wanted or _canonical(_legacy(stored)) in wanted:
            return row
    return None


def cached_path(identity: dict, root: Path | None = None) -> Path | None:
    """Path of a finished JSON for this simulation, when the index already exists."""
    if not index_path(root).is_file():
        return None
    row = find_result(identity, root)
    if row is None:
        return None
    path = _resolve(row["path"], root)
    if path.is_file() and is_finished_result(path):
        return path
    return None


def record_result(path: Path, root: Path | None = None) -> dict | None:
    """Add this JSON to the index if its simulation is not already listed."""
    root = root or results_root()
    row = _row_from_file(path, root)
    if row is None:
        return None
    rows = _rows(root)
    if find_result(row["identity"], root) is not None:
        return row
    rows.append(row)
    _write(root, rows)
    return row


def rebuild(root: Path | None = None) -> list[dict]:
    """Scan ``root`` and keep the first JSON for each simulation."""
    root = (root or results_root()).resolve()
    rows = []
    seen: set[str] = set()
    if not root.is_dir():
        _write(root, rows)
        return rows
    for path in sorted(root.rglob("*.json")):
        if path.name == _NAME or path.name == "budgets.json" or path.name.startswith("selected"):
            continue
        row = _row_from_file(path, root)
        if row is None:
            continue
        key = _canonical(row["identity"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    _write(root, rows)
    return rows


def _rows(root: Path | None) -> list[dict]:
    path = index_path(root)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    simulations = payload.get("simulations") if isinstance(payload, dict) else None
    return list(simulations) if isinstance(simulations, list) else []


def _write(root: Path, rows: list[dict]) -> None:
    path = index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"simulations": rows}, indent=2) + "\n")


def _row_from_file(path: Path, root: Path) -> dict | None:
    identity = _identity_from_file(path)
    if identity is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        stored = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        stored = str(path.resolve())
    return {"identity": identity, "path": stored, "scores": _scores(payload, identity.get("tasks") or [])}


def _scores(payload: dict, tasks: list) -> dict:
    results = payload.get("results") or {}
    children: set[str] = set()
    for kids in (payload.get("group_subtasks") or {}).values():
        children.update(kids or [])
    chosen = [task for task in tasks if task in results]
    if not chosen:
        chosen = [task for task in results if task not in children]
    scores = {}
    for task in chosen:
        metrics = results.get(task) or {}
        if not isinstance(metrics, dict):
            continue
        kept = {key: value for key, value in metrics.items() if key != "alias" and "stderr" not in key}
        if kept:
            scores[task] = kept
    return scores


def _resolve(stored: str, root: Path | None) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return (root or results_root()) / stored
