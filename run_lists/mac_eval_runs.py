"""Small eval run list: Llama 3.2 1B Instruct, sized to fit on a Mac.

Uses CUDA if present, otherwise MPS, otherwise CPU (multi_run.py picks the device).
Edit the loops in config() to add/drop tasks or compression variants.
Results go to mac_eval_results/.
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


def make_run(task, tag, kv, **extra):
    name = f"llama32_1b_{task}_{tag}"
    run = {
        "name": name,
        "tasks": [task],
        "kv": deepcopy(kv),
        "output_path": f"mac_eval_results/{name}.json",
    }
    run.update(deepcopy(extra))
    return run


def config():
    # Nested loops: add another `for` for per-layer or K-only / V-only sweeps.
    # Example: compressions.append((f"k_layer{i}", sparsify([i], [])))
    compressions = (
        ("dense", DENSE),
        ("sparsify48", sparsify("all", "all")),
    )
    runs = []
    for tag, kv in compressions:
        for task, extra in TASKS:
            runs.append(make_run(task, tag, kv, **extra))
    return {
        "model": "hf",
        "batch_size": 1,
        "apply_chat_template": True,
        "num_fewshot": 0,
        "model_args": MODEL_ARGS,
        "runs": runs,
    }
