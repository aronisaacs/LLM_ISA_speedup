#!/usr/bin/env python3
"""WikiText chunk compression on Llama 3.1 8B Instruct.

Fourteen configurations: pooling, top-1, per-tile, and per-feature on keys,
values, and both, with keys undone before RoPE, plus pooling and top-1 on
keys after RoPE. A finished simulation is skipped.

  python scripts/spatial_chunk.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import spatial_feature, spatial_pool, spatial_tile, spatial_top1  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import WIKITEXT_FULL  # noqa: E402
from engine.eval_runner.load_run import grid  # noqa: E402
from engine.eval_runner.progress import mirror_terminal  # noqa: E402

_TARGETS = (
    ("k", {"k_layers": "all", "v_layers": []}),
    ("v", {"k_layers": [], "v_layers": "all"}),
    ("both", {"k_layers": "all", "v_layers": "all"}),
)


def chunk_run() -> dict:
    """The fourteen WikiText configurations."""
    methods = []
    for tag, builder in (
        ("spatial_pool", spatial_pool),
        ("spatial_top1", spatial_top1),
        ("spatial_tile", spatial_tile),
        ("spatial_feature", spatial_feature),
    ):
        methods.extend((f"{tag}_{target}", builder(pre_rope=True, **layers)) for target, layers in _TARGETS)
    methods.extend(
        (
            ("spatial_pool_k_postrope", spatial_pool(k_layers="all", v_layers=[], pre_rope=False)),
            ("spatial_top1_k_postrope", spatial_top1(k_layers="all", v_layers=[], pre_rope=False)),
        )
    )
    return {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
        "configurations": grid(results="figures", methods=methods, tasks=[WIKITEXT_FULL]),
    }


def main() -> None:
    mirror_terminal(ROOT)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(chunk_run(), handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"],
            cwd=ROOT,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
