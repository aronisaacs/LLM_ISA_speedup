"""Read WikiText sweep JSONs into per-(slot, level) perplexity vs dense."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.layer_select.apply import parse_singleton
from engine.layer_select.slots import Slot


@dataclass(frozen=True)
class ScoreRow:
    slot: Slot
    level: int
    ppl: float
    delta: float
    path: str


def word_perplexity(payload: dict) -> float:
    return task_score(payload, "word_perplexity,none")


def task_score(payload: dict, metric: str = "word_perplexity,none") -> float:
    """First task metric in an lm-eval results dict. Higher or lower depends on the task."""
    table = (payload or {}).get("results") or {}
    for metrics in table.values():
        if isinstance(metrics, dict) and metrics.get(metric) is not None:
            return float(metrics[metric])
    raise ValueError(f"no {metric} in results")


def kv_from_payload(payload: dict) -> dict:
    configs = (payload or {}).get("configs") or {}
    for task_config in configs.values():
        if not isinstance(task_config, dict):
            continue
        metadata = task_config.get("metadata") or {}
        if "kv" in metadata:
            return metadata["kv"] or {"pipeline": []}
    model_args = ((payload or {}).get("config") or {}).get("model_args") or {}
    if isinstance(model_args, dict) and "kv" in model_args:
        return model_args["kv"] or {"pipeline": []}
    raise ValueError("no kv metadata in result JSON")


def load_sweep_scores(
    results_dir: str | Path | None = None,
    metric: str = "word_perplexity,none",
    higher_is_better: bool = False,
    *,
    method: str | None = None,
    task: str = "wikitext",
    pretrained: str | None = None,
) -> tuple[float, list[ScoreRow]]:
    """Return (dense score, singleton rows). ``delta`` is quality lost versus dense.

    With no directory, rows come from ``results/index.json``. ``method`` keeps
    only that pipeline (plus the dense baseline).
    """
    if results_dir is None:
        return _from_index(metric, higher_is_better, method=method, task=task, pretrained=pretrained)
    paths = sorted(Path(results_dir).glob("*.json"))
    dense_ppl = None
    rows: list[tuple[Slot, int, float, str]] = []
    for path in paths:
        if path.name.startswith("selected"):
            continue
        payload = json.loads(path.read_text())
        ppl = task_score(payload, metric)
        slot, level = parse_singleton(kv_from_payload(payload))
        if slot is None:
            dense_ppl = ppl
            continue
        if level is None:
            raise ValueError(f"{path} singleton has no compression level")
        rows.append((slot, level, ppl, str(path)))
    if dense_ppl is None:
        raise ValueError(f"no dense result in {results_dir}")
    return dense_ppl, _scored(dense_ppl, rows, higher_is_better)


def _from_index(metric, higher_is_better, method, task, pretrained) -> tuple[float, list[ScoreRow]]:
    from engine.eval_runner.index import results_root, simulations

    root = results_root()
    dense_ppl = None
    rows: list[tuple[Slot, int, float, str]] = []
    for record in simulations():
        identity = record.get("identity") or {}
        if task not in (identity.get("tasks") or []):
            continue
        if pretrained is not None and identity.get("pretrained") != pretrained:
            continue
        kv = identity.get("kv") or {}
        pipeline = kv.get("pipeline") or []
        if pipeline and method is not None and pipeline[0].get("method") != method:
            continue
        try:
            slot, level = parse_singleton(kv)
        except ValueError:
            continue
        stored = root / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"])
        value = (record.get("scores") or {}).get(task, {}).get(metric)
        if value is None:
            continue
        ppl = float(value)
        if slot is None:
            if pipeline:
                continue
            dense_ppl = ppl
            continue
        if level is None:
            raise ValueError(f"{stored} singleton has no compression level")
        rows.append((slot, level, ppl, str(stored)))
    if dense_ppl is None:
        raise ValueError(f"no dense {task} result in the index")
    return dense_ppl, _scored(dense_ppl, rows, higher_is_better)


def _scored(dense_ppl, rows, higher_is_better) -> list[ScoreRow]:
    return [
        ScoreRow(
            slot=slot,
            level=level,
            ppl=ppl,
            delta=(dense_ppl - ppl) if higher_is_better else (ppl - dense_ppl),
            path=path,
        )
        for slot, level, ppl, path in rows
    ]


def score_table(rows: list[ScoreRow]) -> dict[tuple[Slot, int], ScoreRow]:
    return {(row.slot, row.level): row for row in rows}
