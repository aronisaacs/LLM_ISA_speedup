"""Run: for each compression (dense, all-layer 4:8), for each task.

Currently Llama 3.2 1B Instruct, ARC-Easy and GSM8K with short limits.
Results go to results/compression_x_task/. Device is CUDA / MPS / CPU.
"""

from __future__ import annotations

from copy import deepcopy

from catalog.compressions import DENSE, SPARSIFY_48
from catalog.models import LLAMA32_1B
from catalog.tasks import ARC_EASY_256, GSM8K_32


def make_configuration(task_name, tag, kv, **extra):
    name = f"llama32_1b_{task_name}_{tag}"
    configuration = {
        "name": name,
        "tasks": [task_name],
        "kv": deepcopy(kv),
        "output_path": f"results/compression_x_task/{name}.json",
    }
    configuration.update(deepcopy(extra))
    return configuration


def run():
    # One configuration per compression and task.
    compressions = (
        ("dense", DENSE),
        ("sparsify48", SPARSIFY_48),
    )
    tasks = (
        ("arc_easy", {key: value for key, value in ARC_EASY_256.items() if key != "tasks"}),
        ("gsm8k", {key: value for key, value in GSM8K_32.items() if key != "tasks"}),
    )
    configurations = []
    for tag, kv in compressions:
        for task_name, extra in tasks:
            configurations.append(make_configuration(task_name, tag, kv, **extra))
    return {
        "model": "hf",
        "batch_size": 1,
        "apply_chat_template": True,
        "num_fewshot": 0,
        "model_args": LLAMA32_1B,
        "configurations": configurations,
    }
