"""Run: for each compression (dense, all-layer 4:8), CEval 5-shot.

Llama 3.1 8B Instruct, full ceval-valid. Matches Mustafa's all-layer 4:8 CEval table
(~55% dense, ~37% 4:8). Results go to results/compression_x_ceval/.
"""

from __future__ import annotations

from copy import deepcopy

from catalog.compressions import DENSE, SPARSIFY_48
from catalog.models import LLAMA31_8B


def make_configuration(tag, kv):
    name = f"llama31_ceval_{tag}"
    return {
        "name": name,
        "tasks": ["ceval-valid"],
        "kv": deepcopy(kv),
        "output_path": f"results/compression_x_ceval/{name}.json",
    }


def run():
    compressions = (
        ("dense", DENSE),
        ("sparsify48", SPARSIFY_48),
    )
    configurations = [make_configuration(tag, kv) for tag, kv in compressions]
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "apply_chat_template": True,
        "num_fewshot": 5,
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }
