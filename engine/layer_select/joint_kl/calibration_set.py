"""The KL calibration manifest: load, verify, and split across workers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ("wikitext", "arc_5shot", "gsm8k", "mbpp")
LLAMA31_MIXED4 = ROOT / "catalog" / "calibration" / "llama31_8b_mixed4"


def load(directory: str | Path = LLAMA31_MIXED4, per_domain: int | None = None) -> tuple[list[dict], dict]:
    """Rows and metadata. The manifest must match the SHA-256 in its metadata.

    ``per_domain`` keeps the first that many examples of each domain (timing runs).
    """
    directory = Path(directory)
    meta = json.loads((directory / "manifest_metadata.json").read_text())
    manifest = directory / "manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != meta["manifest_sha256"]:
        raise ValueError(f"{manifest} does not match manifest_metadata.json ({digest[:12]})")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if [row["position"] for row in rows] != list(range(len(rows))):
        raise ValueError("manifest positions are not 0..N-1")
    unknown = {row["domain"] for row in rows} - set(DOMAINS)
    if unknown:
        raise ValueError(f"unknown domains {sorted(unknown)}")
    if per_domain is not None:
        seen = {domain: 0 for domain in DOMAINS}
        kept = []
        for row in rows:
            if seen[row["domain"]] < per_domain:
                seen[row["domain"]] += 1
                kept.append(row)
        rows = kept
    return rows, meta


def cost(row: dict) -> int:
    """Forward length, a proxy for the time one example takes."""
    return len(row["prompt_token_ids"]) + len(row["reference_token_ids"]) - 1


def partition(rows: list[dict], workers: int) -> list[list[dict]]:
    """Longest example first to the least loaded worker, so all workers finish together."""
    if workers < 1:
        raise ValueError("need at least one worker")
    shards: list[list[dict]] = [[] for _ in range(workers)]
    load_by = [0] * workers
    for row in sorted(rows, key=lambda r: (-cost(r), r["position"])):
        index = min(range(workers), key=lambda w: (load_by[w], w))
        shards[index].append(row)
        load_by[index] += cost(row)
    return [sorted(shard, key=lambda r: r["position"]) for shard in shards]
