"""Run: for each compression (dense, all-layer 4:8), for each task.

Currently Llama 3.2 1B Instruct, ARC-Easy and GSM8K with short limits.
Results go to compression_x_task_results/. Device is CUDA / MPS / CPU.
"""

from __future__ import annotations

from copy import deepcopy

DENSE = {"pipeline": []}

MODEL_ARGS = "pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype=float16"

TASKS = (
    ("arc_easy", {"limit": 256}),
    ("gsm8k", {"limit": 32, "gen_kwargs": {"max_gen_toks": 128}}),
)


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


def make_configuration(task, tag, kv, **extra):
    name = f"llama32_1b_{task}_{tag}"
    configuration = {
        "name": name,
        "tasks": [task],
        "kv": deepcopy(kv),
        "output_path": f"compression_x_task_results/{name}.json",
    }
    configuration.update(deepcopy(extra))
    return configuration


def run():
    # Nested loops: add another `for` for per-layer or K-only / V-only sweeps.
    # Example: compressions.append((f"k_layer{i}", sparsify([i], [])))
    compressions = (
        ("dense", DENSE),
        ("sparsify48", sparsify("all", "all")),
    )
    configurations = []
    for tag, kv in compressions:
        for task, extra in TASKS:
            configurations.append(make_configuration(task, tag, kv, **extra))
    return {
        "model": "hf",
        "batch_size": 1,
        "apply_chat_template": True,
        "num_fewshot": 0,
        "model_args": MODEL_ARGS,
        "configurations": configurations,
    }
