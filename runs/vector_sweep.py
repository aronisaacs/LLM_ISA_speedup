"""WikiText singleton sweep of vector compression on Llama 3.1 8B Instruct.

Dense, then each key and each value at 25%, 50%, and 75%.
Results go to results/vector_sweep/. The study driver runs this first.
"""

from __future__ import annotations

from catalog.compressions import vector_compress
from catalog.models import LLAMA31_8B
from catalog.tasks import WIKITEXT_FULL
from layer_select.budgets import LLAMA31_LAYERS, SWEEP_RESULTS
from layer_select.sweep import expand_singleton_configs


def run():
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=vector_compress(prune_pct=25),
        method_tag="vector",
        results_dir=SWEEP_RESULTS,
        name_prefix="llama31",
        extra=WIKITEXT_FULL,
    )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }
