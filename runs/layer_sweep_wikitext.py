"""Run: WikiText singleton sweep over slots × compression rungs.

Llama 3.2 1B, checksparse L1, keys-only, rungs 15/30/40/50/60%. Small limit.
Results go to results/layer_sweep_wikitext/.
"""

from __future__ import annotations

from catalog.compressions import checksparse_l1
from catalog.models import LLAMA32_1B
from catalog.tasks import WIKITEXT
from layer_select.levels import DEFAULT_LEVELS
from layer_select.slots import n_hidden_layers
from layer_select.sweep import expand_singleton_configs

SPACE = "keys_only"
METHOD_TAG = "checksparse"
METHOD_KV = checksparse_l1("all", "all")
LEVELS = DEFAULT_LEVELS
RESULTS_DIR = "results/layer_sweep_wikitext"
NAME_PREFIX = "llama32_1b_wikitext"


def run():
    extra = dict(WIKITEXT)
    configurations = expand_singleton_configs(
        n_layers=n_hidden_layers(LLAMA32_1B),
        space=SPACE,
        method_kv=METHOD_KV,
        method_tag=METHOD_TAG,
        results_dir=RESULTS_DIR,
        name_prefix=NAME_PREFIX,
        extra=extra,
        levels=LEVELS,
    )
    return {
        "model": "hf",
        "batch_size": 1,
        "model_args": LLAMA32_1B,
        "configurations": configurations,
    }
