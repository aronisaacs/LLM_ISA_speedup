"""One row per finished simulation.

``results/index.json`` is the record: identity, scores, sample count, and,
for a budget eval, the target budget and the realized compression.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.eval_runner.cache import _canonical, _identity_from_file, _legacy

_NAME = "index.json"


def results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def index_path(root: Path | None = None) -> Path:
    return (root or results_root()) / _NAME


def simulations(root: Path | None = None) -> list[dict]:
    """Every finished simulation in the index."""
    return _rows(root)


def find_result(identity: dict, root: Path | None = None) -> dict | None:
    """The index row for this simulation, or ``None`` when it has not been scored."""
    rows = _rows(root)
    wanted = {_canonical(identity), _canonical(_legacy(identity))}
    for row in rows:
        stored = row.get("identity") or {}
        if _canonical(stored) in wanted or _canonical(_legacy(stored)) in wanted:
            return row
    return None


def record_simulation(
    identity: dict,
    scores: dict,
    samples: dict | None = None,
    budget: float | None = None,
    compression: float | None = None,
    root: Path | None = None,
) -> dict:
    """Append this simulation when the index does not already list it."""
    root = root or results_root()
    existing = find_result(identity, root)
    if existing is not None:
        return existing
    row = {"identity": identity, "scores": scores}
    if samples:
        row["samples"] = samples
    if budget is not None:
        row["budget"] = budget
    if compression is not None:
        row["compression"] = compression
    rows = _rows(root)
    rows.append(row)
    _write(root, rows)
    return row


def record_result(path: Path, root: Path | None = None) -> dict | None:
    """Add a result JSON to the index if its simulation is not already listed."""
    root = root or results_root()
    row = _row_from_file(path)
    if row is None:
        return None
    return record_simulation(
        row["identity"],
        row["scores"],
        samples=row.get("samples"),
        budget=row.get("budget"),
        compression=row.get("compression"),
        root=root,
    )


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
        row = _row_from_file(path)
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


def _row_from_file(path: Path) -> dict | None:
    identity = _identity_from_file(path)
    if identity is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    row = {"identity": identity, "scores": _scores(payload, identity.get("tasks") or [])}
    samples = _samples(payload)
    if samples:
        row["samples"] = samples
    budget = _budget(payload)
    if budget is not None:
        row["budget"] = budget[0]
        row["compression"] = budget[1]
    return row


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


def _samples(payload: dict) -> dict | None:
    raw = payload.get("n-samples") or {}
    for info in raw.values():
        if isinstance(info, dict) and ("effective" in info or "original" in info):
            return {key: info[key] for key in ("original", "effective") if key in info}
    return None


def _budget(payload: dict) -> tuple[float, float] | None:
    for task_config in (payload.get("configs") or {}).values():
        metadata = (task_config or {}).get("metadata") or {}
        if "kv_budget" in metadata and "kv_compression" in metadata:
            return float(metadata["kv_budget"]), float(metadata["kv_compression"])
    return None


