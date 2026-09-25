"""Llama 3.1 8B WikiText quality runs.

  python engine/multi_run.py --run runs/llama31.py:qjl
  python engine/multi_run.py --run runs/llama31.py:spatial_chunk
"""

from __future__ import annotations

from catalog.compressions import qjl as qjl_kv, spatial_feature, spatial_pool, spatial_tile, spatial_top1
from catalog.models import LLAMA31_8B
from catalog.tasks import WIKITEXT_FULL
from engine.eval_runner.load_run import grid

_TARGETS = (
    ("k", {"k_layers": "all", "v_layers": []}),
    ("v", {"k_layers": [], "v_layers": "all"}),
    ("both", {"k_layers": "all", "v_layers": "all"}),
)


def qjl():
    """Hadamard codebook on keys, values, and both. Post-RoPE. Scores in results/qjl/."""
    methods = [(f"qjl_{tag}", qjl_kv(**layers)) for tag, layers in _TARGETS]
    return _llama31("results/qjl", methods)


def spatial_chunk():
    """Four chunk methods on keys, values, and both, plus two post-RoPE key controls.

    Scores go to results/spatial_chunk/.
    """
    methods = []
    for tag, builder in (
        ("spatial_pool", spatial_pool),
        ("spatial_top1", spatial_top1),
        ("spatial_tile", spatial_tile),
        ("spatial_feature", spatial_feature),
    ):
        methods.extend((f"{tag}_{target}", builder(pre_rope=True, **layers)) for target, layers in _TARGETS)
    controls = [
        ("spatial_pool_k_postrope", spatial_pool(k_layers="all", v_layers=[], pre_rope=False)),
        ("spatial_top1_k_postrope", spatial_top1(k_layers="all", v_layers=[], pre_rope=False)),
    ]
    return _llama31("results/spatial_chunk", methods + controls)


def _llama31(results: str, methods) -> dict:
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": grid(results=results, methods=methods, tasks=[WIKITEXT_FULL]),
    }
