"""Dense versus all-layer 4:8 comparisons.

  python engine/multi_run.py --run runs/comparisons.py:compression_x_task
  python engine/multi_run.py --run runs/comparisons.py:compression_x_ceval
  python engine/multi_run.py --run runs/comparisons.py:model_x_compression_x_task
"""

from __future__ import annotations

from catalog.compressions import DENSE, SPARSIFY_48
from catalog.models import CODELLAMA_7B, LLAMA31_8B, LLAMA32_1B, QWEN35_9B
from catalog.tasks import (
    ARC_EASY_256,
    CEVAL_VALID_5SHOT,
    GSM8K_20PCT,
    GSM8K_32,
    HUMANEVAL_CODELLAMA,
    HUMANEVAL_INSTRUCT,
)
from engine.eval_runner.load_run import grid

_SPARSE = (
    ("dense", DENSE),
    ("sparsify48", SPARSIFY_48),
)

_MODELS = (
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


def compression_x_task():
    """Llama 3.2 1B, ARC-Easy and a short GSM8K, dense then 4:8."""
    return {
        "model": "hf",
        "batch_size": 1,
        "apply_chat_template": True,
        "num_fewshot": 0,
        "model_args": LLAMA32_1B,
        "configurations": grid(
            results="results/compression_x_task",
            methods=_SPARSE,
            tasks=(ARC_EASY_256, GSM8K_32),
            model_tag="llama32_1b",
        ),
    }


def compression_x_ceval():
    """Llama 3.1 8B, full ceval-valid, dense then 4:8."""
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "apply_chat_template": True,
        "num_fewshot": 5,
        "model_args": LLAMA31_8B,
        "configurations": grid(
            results="results/compression_x_ceval",
            methods=_SPARSE,
            tasks=({"name_task": "ceval", "tasks": ["ceval-valid"]},),
        ),
    }


def model_x_compression_x_task():
    """Llama 3.1, Qwen 3.5, and CodeLlama. Compression stays inside each model."""
    configurations = []
    for model in _MODELS:
        configurations.extend(
            grid(
                results=model["out"],
                methods=_SPARSE,
                tasks=model["tasks"],
                model_tag=model["id"],
                model_args=model["model_args"],
            )
        )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "configurations": configurations,
    }
