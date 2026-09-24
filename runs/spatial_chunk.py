"""WikiText on Llama 3.1 8B for the four static chunk methods.

Each method on keys, values, and both, with keys undone before RoPE.
Two controls compress post-RoPE keys: pooling and top-1.
Scores go to results/spatial_chunk/.
"""

from __future__ import annotations

from copy import deepcopy

from catalog.compressions import spatial_feature, spatial_pool, spatial_tile, spatial_top1
from catalog.models import LLAMA31_8B
from catalog.tasks import WIKITEXT_FULL

RESULTS = "results/spatial_chunk"


def _configuration(tag, kv):
    extra = {key: value for key, value in WIKITEXT_FULL.items() if key != "tasks"}
    return {
        "name": f"llama31_{tag}",
        "tasks": ["wikitext"],
        "kv": deepcopy(kv),
        "output_path": f"{RESULTS}/llama31_{tag}.json",
        **extra,
    }


def run():
    methods = (
        ("spatial_pool", spatial_pool),
        ("spatial_top1", spatial_top1),
        ("spatial_tile", spatial_tile),
        ("spatial_feature", spatial_feature),
    )
    targets = (
        ("k", {"k_layers": "all", "v_layers": []}),
        ("v", {"k_layers": [], "v_layers": "all"}),
        ("both", {"k_layers": "all", "v_layers": "all"}),
    )
    configurations = []
    for tag, builder in methods:
        for target, layers in targets:
            configurations.append(_configuration(f"{tag}_{target}", builder(pre_rope=True, **layers)))
    for tag, builder in (("spatial_pool", spatial_pool), ("spatial_top1", spatial_top1)):
        configurations.append(
            _configuration(f"{tag}_k_postrope", builder(k_layers="all", v_layers=[], pre_rope=False))
        )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }
