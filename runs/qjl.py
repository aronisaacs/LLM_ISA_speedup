"""WikiText on Llama 3.1 8B for Hadamard codebook keys, values, and both.

Each token is rotated, snapped to a 4-bit grid, and rescaled by an fp16 norm.
Scores go to results/qjl/.
"""

from __future__ import annotations

from copy import deepcopy

from catalog.compressions import qjl
from catalog.models import LLAMA31_8B
from catalog.tasks import WIKITEXT_FULL

RESULTS = "results/qjl"


def run():
    extra = {key: value for key, value in WIKITEXT_FULL.items() if key != "tasks"}
    targets = (
        ("k", {"k_layers": "all", "v_layers": []}),
        ("v", {"k_layers": [], "v_layers": "all"}),
        ("both", {"k_layers": "all", "v_layers": "all"}),
    )
    configurations = []
    for tag, layers in targets:
        configurations.append(
            {
                "name": f"llama31_qjl_{tag}",
                "tasks": ["wikitext"],
                "kv": deepcopy(qjl(**layers)),
                "output_path": f"{RESULTS}/llama31_qjl_{tag}.json",
                **extra,
            }
        )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }
