"""Run: for each model, for each compression (dense, all-layer 4:8), for each task.

Currently Llama 3.1 8B, Qwen 3.5 9B, CodeLlama 7B. Compression inner so the
checkpoint load is reused. Device is CUDA / MPS / CPU.
"""

from __future__ import annotations

from copy import deepcopy

from catalog.compressions import DENSE, SPARSIFY_48
from catalog.models import CODELLAMA_7B, LLAMA31_8B, QWEN35_9B
from catalog.tasks import CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_CODELLAMA, HUMANEVAL_INSTRUCT


MODELS = (
    {
        "id": "llama31",
        "out": "results/model_x_compression_x_task/llama3",
        "model_args": LLAMA31_8B,
        "tasks": (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT),
    },
    {
        "id": "qwen35",
        "out": "results/model_x_compression_x_task/qwen",
        "model_args": QWEN35_9B,
        "tasks": (HUMANEVAL_INSTRUCT, GSM8K_20PCT, CEVAL_VALID_5SHOT),
    },
    {
        "id": "codellama7b",
        "out": "results/model_x_compression_x_task/codellama",
        "model_args": CODELLAMA_7B,
        "tasks": (HUMANEVAL_CODELLAMA,),
    },
)


def make_configuration(model, task, tag, kv):
    extra = {key: value for key, value in task.items() if key not in {"name_task", "file"}}
    configuration = {
        "name": f"{model['id']}_{task['name_task']}_{tag}",
        "model_args": model["model_args"],
        "kv": deepcopy(kv),
        "output_path": f"{model['out']}/{task['file']}_{tag}.json",
    }
    configuration.update(deepcopy(extra))
    return configuration


def run():
    compressions = (
        ("dense", DENSE),
        ("sparsify48", SPARSIFY_48),
    )
    configurations = []
    for model in MODELS:
        for tag, kv in compressions:
            for task in model["tasks"]:
                configurations.append(make_configuration(model, task, tag, kv))
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "configurations": configurations,
    }
