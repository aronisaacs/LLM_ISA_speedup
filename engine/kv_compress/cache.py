"""Route cache updates through registered compression and after-append hooks.

Methods own reconstruction and completion of groups spanning updates. The cache
owns dispatch, layer selection, absolute positions and dense-byte observations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.kv_compress.methods import AFTER_APPEND
from engine.kv_compress import metrics
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables
from engine.kv_compress.spec import KvSpec, LayerSelection

_original_update: Callable[..., Any] | None = None


def patch_cache_update(spec: KvSpec, rope: RopeTables | None = None, *, take_query=None) -> Callable[[], None]:
    """Wrap ``Cache.update`` so new K/V pass through ``spec`` before they are stored.

    ``rope`` is forwarded to each compression method. Returns a zero-arg uninstall function.
    """
    from transformers.cache_utils import Cache

    global _original_update
    saved = _original_update
    if saved is None:
        saved = Cache.update
        _original_update = saved
    previous = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        start = _seq_length(self, layer_idx)
        end = start + key_states.shape[-2]
        token_weights = None
        query = take_query() if take_query is not None else None
        needs_importance = any(step.kwargs.get('importance') == 'prefill_attention' and
                               (_enabled(step.k_layers, layer_idx) or _enabled(step.v_layers, layer_idx))
                               for step in spec.pipeline)
        if needs_importance and start == 0:
            if query is None:
                raise ValueError('missing current prefill queries for token importance')
            from engine.kv_compress.importance import attention_received
            token_weights = attention_received(query, key_states)[0].unsqueeze(0)
            token_weights = token_weights / token_weights.mean().clamp_min(1e-12)
        metrics.observe(key_states, value_states, layer_idx, spec)
        key_states, value_states = compress_kv(
            key_states, value_states, layer_idx, spec, seq_start=start, rope=rope,
            token_weights=token_weights
        )
        keys, values = saved(self, key_states, value_states, layer_idx, *args, **kwargs)
        _after_append(keys, values, layer_idx, spec, start, end, rope)
        return keys, values

    Cache.update = update

    def uninstall() -> None:
        Cache.update = previous

    return uninstall


def _after_append(keys, values, layer_idx, spec, start, end, rope):
    for step in spec.pipeline:
        callback = AFTER_APPEND.get(step.method)
        if callback is None:
            continue
        for target, tensor, selection in (("k", keys, step.k_layers), ("v", values, step.v_layers)):
            if _enabled(selection, layer_idx) and tensor.shape[-2] >= end:
                callback(tensor, target=target, layer_idx=layer_idx, start=start,
                         end=end, rope_tables=rope, **step.kwargs)


def _enabled(selection: LayerSelection, layer_idx: int) -> bool:
    if selection == "all":
        return True
    return layer_idx in selection


def _seq_length(cache, layer_idx: int) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if getter is None:
        return 0
    try:
        return int(getter(layer_idx))
    except TypeError:
        return int(getter())
