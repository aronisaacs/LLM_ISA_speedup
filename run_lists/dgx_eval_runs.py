"""Lab-scale eval run list: Llama 3.1 8B, Qwen 3.5 9B, CodeLlama 7B.

Dense then 4:8 sparsify per model so the checkpoint load is reused.
Fits a DGX; too large for a Mac. Device is CUDA/MPS/CPU from multi_run.py.
"""

from __future__ import annotations

from copy import deepcopy

DENSE = {"pipeline": []}

CEVAL = {
    "name_task": "ceval",
    "file": "ceval",
    "tasks": ["ceval-valid"],
    "num_fewshot": 5,
    "apply_chat_template": True,
}
GSM8K = {
    "name_task": "gsm8k",
    "file": "gsm8k_20pct",
    "tasks": ["gsm8k"],
    "samples": "@gsm8k_samples_profile20pct.json",
    "num_fewshot": 0,
    "gen_kwargs": {"max_gen_toks": 512},
    "apply_chat_template": True,
}
HUMANEVAL_INSTRUCT = {
    "name_task": "humaneval_instruct",
    "file": "humaneval_instruct",
    "tasks": ["humaneval_instruct"],
    "num_fewshot": 0,
    "env": {"HF_ALLOW_CODE_EVAL": "1"},
    "confirm_run_unsafe_code": True,
    "apply_chat_template": True,
}
HUMANEVAL_CODELLAMA = {
    "name_task": "humaneval",
    "file": "humaneval",
    "tasks": ["humaneval_codellama"],
    "num_fewshot": 0,
    "env": {"HF_ALLOW_CODE_EVAL": "1"},
    "confirm_run_unsafe_code": True,
}

MODELS = (
    {
        "id": "llama31",
        "out": "llama3_test",
        "model_args": "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16",
        "tasks": (CEVAL, GSM8K, HUMANEVAL_INSTRUCT),
    },
    {
        "id": "qwen35",
        "out": "qwen_test",
        "model_args": "pretrained=Qwen/Qwen3.5-9B,dtype=bfloat16,enable_thinking=False",
        "tasks": (HUMANEVAL_INSTRUCT, GSM8K, CEVAL),
    },
    {
        "id": "codellama7b",
        "out": "codellama_test",
        "model_args": "pretrained=meta-llama/CodeLlama-7b-hf,dtype=bfloat16",
        "tasks": (HUMANEVAL_CODELLAMA,),
    },
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


def make_run(model, task, tag, kv):
    extra = {key: value for key, value in task.items() if key not in {"name_task", "file"}}
    run = {
        "name": f"{model['id']}_{task['name_task']}_{tag}",
        "model_args": model["model_args"],
        "kv": deepcopy(kv),
        "output_path": f"{model['out']}/{task['file']}_{tag}.json",
    }
    run.update(deepcopy(extra))
    return run


def config():
    compressions = (
        ("dense", DENSE),
        ("sparsify48", sparsify("all", "all")),
    )
    runs = []
    for model in MODELS:
        for tag, kv in compressions:
            for task in model["tasks"]:
                runs.append(make_run(model, task, tag, kv))
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "runs": runs,
    }
