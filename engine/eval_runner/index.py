"""One row per finished simulation.

``results.json`` is the record: identity, scores, sample count, and,
for a budget eval, separate target, planned and measured compression.
Historical rows retain their original `compression` field; it is not relabeled
as a measurement. Writes are locked across processes and replaced atomically.
"""

from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
from pathlib import Path

from engine.eval_runner.cache import _canonical, _identity_from_file, identities_match

_NAME = "results.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def index_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / _NAME


def simulations(root: Path | None = None) -> list[dict]:
    """Every finished simulation in the index."""
    return _rows(root)


def find_result(identity: dict, root: Path | None = None) -> dict | None:
    """The index row for this simulation, or ``None`` when it has not been scored."""
    rows = _rows(root)
    for row in rows:
        stored = row.get("identity") or {}
        if identities_match(stored, identity):
            return row
    return None


def drop_simulations(predicate, root: Path | None = None) -> int:
    """Remove index rows for which ``predicate`` is true. Returns how many dropped."""
    root = root or repo_root()
    with _locked(root):
        rows = _rows(root)
        kept = [row for row in rows if not predicate(row)]
        removed = len(rows) - len(kept)
        if removed:
            _write(root, kept)
        return removed


def record_simulation(
    identity: dict,
    scores: dict,
    samples: dict | None = None,
    budget: float | None = None,
    compression: float | None = None,
    root: Path | None = None,
    *, planned_compression: float | None = None, storage: dict | None = None,
    compression_target: str = "kv",
) -> dict:
    """Append this simulation when the index does not already list it."""
    root = root or repo_root()
    with _locked(root):
        existing = find_result(identity, root)
        if existing is not None:
            return existing
        row = {"identity": identity, "scores": scores}
        if samples:
            row["samples"] = samples
        if budget is not None:
            row["budget"] = budget
        if compression is not None or planned_compression is not None:
            # Historical callers supply planned savings as `compression`.
            row["planned_compression"] = planned_compression if planned_compression is not None else compression
        if storage:
            row["storage"] = storage
            row["compression_target"] = compression_target
            row["measured_compression"] = storage["targets"][compression_target]["compression"]
        rows = _rows(root)
        rows.append(row)
        _write(root, rows)
        return row


def record_result(path: Path, root: Path | None = None) -> dict | None:
    """Add a result JSON to the index if its simulation is not already listed."""
    root = root or repo_root()
    row = _row_from_file(path)
    if row is None:
        return None
    return record_simulation(
        row["identity"],
        row["scores"],
        samples=row.get("samples"),
        budget=row.get("budget"),
        planned_compression=row.get("planned_compression"),
        storage=row.get("storage"),
        compression_target=row.get("compression_target", "kv"),
        root=root,
    )


def rebuild(root: Path) -> list[dict]:
    """Import detailed result files, preserving scores already in the ledger."""
    root = root.resolve()
    with _locked(root):
        rows = _rows(root)
        seen = {_canonical(row["identity"]) for row in rows}
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
    payload = json.loads(path.read_text())
    rows = payload.get("simulations") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid results ledger: {path}")
    return rows


@contextmanager
def _locked(root):
    root.mkdir(parents=True, exist_ok=True)
    # Keep a stable lock inode: never unlink the lock file after releasing it.
    with (root / ".results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _write(root: Path, rows: list[dict]) -> None:
    from engine.eval_runner.files import write_json
    write_json(index_path(root), {"simulations": rows})


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
        row["planned_compression"] = budget[1]
    if payload.get("storage"):
        row["storage"] = payload["storage"]
        metadata = next((c.get("metadata", {}) for c in payload.get("configs", {}).values()
                         if c.get("metadata", {}).get("compression_target")), {})
        target = metadata.get("compression_target", "kv")
        row["compression_target"] = target
        row["measured_compression"] = payload["storage"]["targets"][target]["compression"]
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


