"""Run: for each compression (dense, all-layer 4:8), CEval 5-shot.

Llama 3.1 8B Instruct, full ceval-valid. Matches Mustafa's all-layer 4:8 CEval table
(~55% dense, ~37% 4:8). Results go to results/compression_x_ceval/.
"""

from __future__ import annotations

from copy import deepcopy

DENSE = {"pipeline": []}

MODEL_ARGS = "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16"


def sparsify(k_layers, v_layers, n=8, m=4):
    return {
        "pipeline": [
            {
                "method": "sparsify_nm",
                "n": n,
                "m": m,
                "k_layers": k_layers,
                "v_layers": v_layers,
            }
        ]
    }


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
        ("sparsify48", sparsify("all", "all")),
    )
    configurations = [make_configuration(tag, kv) for tag, kv in compressions]
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "apply_chat_template": True,
        "num_fewshot": 5,
        "model_args": MODEL_ARGS,
        "configurations": configurations,
    }
