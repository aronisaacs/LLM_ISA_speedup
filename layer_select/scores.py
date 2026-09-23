"""Read WikiText sweep JSONs into per-(slot, level) perplexity vs dense."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from layer_select.apply import parse_singleton
from layer_select.slots import Slot


@dataclass(frozen=True)
class ScoreRow:
    slot: Slot
    level: int
    ppl: float
    delta: float
    path: str


def word_perplexity(payload: dict) -> float:
    table = (payload or {}).get("results") or {}
    for metrics in table.values():
        if isinstance(metrics, dict) and metrics.get("word_perplexity,none") is not None:
            return float(metrics["word_perplexity,none"])
    raise ValueError("no word_perplexity,none in results")


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


def load_sweep_scores(results_dir: str | Path) -> tuple[float, list[ScoreRow]]:
    """Return (dense_ppl, singleton rows with delta = ppl - dense)."""
    paths = sorted(Path(results_dir).glob("*.json"))
    dense_ppl = None
    rows: list[tuple[Slot, int, float, str]] = []
    for path in paths:
        if path.name.startswith("selected"):
            continue
        payload = json.loads(path.read_text())
        ppl = word_perplexity(payload)
        slot, level = parse_singleton(kv_from_payload(payload))
        if slot is None:
            dense_ppl = ppl
            continue
        if level is None:
            raise ValueError(f"{path} singleton has no compression level")
        rows.append((slot, level, ppl, str(path)))
    if dense_ppl is None:
        raise ValueError(f"no dense result in {results_dir}")
    scored = [
        ScoreRow(slot=slot, level=level, ppl=ppl, delta=ppl - dense_ppl, path=path)
        for slot, level, ppl, path in rows
    ]
    return dense_ppl, scored


def score_table(rows: list[ScoreRow]) -> dict[tuple[Slot, int], ScoreRow]:
    return {(row.slot, row.level): row for row in rows}
