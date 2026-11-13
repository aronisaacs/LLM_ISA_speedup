"""Route cache updates through registered compression and after-append hooks.

Methods own reconstruction and completion of groups spanning updates. The cache
owns dispatch, layer selection, absolute positions and dense-byte observations.

A step with ``recent_window`` W > 0 (only methods that compress each vector on its own accept
it) leaves the newest W tokens exact: the cache stores new tokens as they are, and compresses a
token in place once it is W tokens old. A method that compresses groups of G tokens together
(``WINDOW_GROUPS``) ages them out in whole aligned groups, so W to W + G - 1 tokens stay exact. The prompt's own attention therefore sees its last W
tokens exact and older ones compressed, unless ``prompt_attention`` is "dense".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.kv_compress.methods import AFTER_APPEND, WINDOW_GROUPS, get_method
from engine.kv_compress import metrics
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables
from engine.kv_compress.spec import KvSpec, LayerSelection

_original_update: Callable[..., Any] | None = None


def patch_cache_update(spec: KvSpec, rope: RopeTables | None = None) -> Callable[[], None]:
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

    windowed = tuple(step for step in spec.pipeline if step.kwargs.get("recent_window", 0))
    immediate = KvSpec(pipeline=tuple(step for step in spec.pipeline if step not in windowed),
                       prompt_attention=spec.prompt_attention)

    def update_windowed(self, key_states, value_states, layer_idx, *args, **kwargs):
        start = _seq_length(self, layer_idx)
        end = start + key_states.shape[-2]
        metrics.observe(key_states, value_states, layer_idx, spec)
        exact = None
        if _dense_prompt(spec, start, key_states):
            exact = (key_states.clone(), value_states.clone())
        key_states, value_states = compress_kv(key_states, value_states, layer_idx, immediate, seq_start=start,
                                               rope=rope)
        keys, values = saved(self, key_states, value_states, layer_idx, *args, **kwargs)
        for step in windowed:
            _compress_aged(step, keys, values, layer_idx, start, end, rope)
        _after_append(keys, values, layer_idx, spec, start, end, rope)
        return exact if exact is not None else (keys, values)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        start = _seq_length(self, layer_idx)
        end = start + key_states.shape[-2]
        metrics.observe(key_states, value_states, layer_idx, spec)
        exact = (key_states, value_states)
        key_states, value_states = compress_kv(key_states, value_states, layer_idx, spec, seq_start=start, rope=rope)
        keys, values = saved(self, key_states, value_states, layer_idx, *args, **kwargs)
        _after_append(keys, values, layer_idx, spec, start, end, rope)
        if _dense_prompt(spec, start, exact[0]):
            # The cache keeps the compressed prompt; only this first call attends to it exactly.
            return exact
        return keys, values

    Cache.update = update_windowed if windowed else update

    def uninstall() -> None:
        Cache.update = previous

    return uninstall


def _compress_aged(step, keys, values, layer_idx, start, end, rope):
    """Compress, in place in the stored cache, the tokens that left this step's window in this update."""
    window = step.kwargs["recent_window"]
    group = WINDOW_GROUPS[step.method](step.kwargs) if step.method in WINDOW_GROUPS else 1
    first, last = max(start - window, 0) // group * group, max(end - window, 0) // group * group
    if last <= first:
        return
    method = get_method(step.method)
    for target, tensor, selection in (("k", keys, step.k_layers), ("v", values, step.v_layers)):
        if _enabled(selection, layer_idx):
            aged = tensor[..., first:last, :]
            tensor[..., first:last, :] = method(aged, layer_idx=layer_idx, target=target, seq_start=first,
                                                rope_tables=rope, **step.kwargs)


def _after_append(keys, values, layer_idx, spec, start, end, rope):
    for step in spec.pipeline:
        callback = AFTER_APPEND.get(step.method)
        if callback is None:
            continue
        for target, tensor, selection in (("k", keys, step.k_layers), ("v", values, step.v_layers)):
            if _enabled(selection, layer_idx) and tensor.shape[-2] >= end:
                callback(tensor, target=target, layer_idx=layer_idx, start=start,
                         end=end, rope_tables=rope, **step.kwargs)


def _dense_prompt(spec: KvSpec, start: int, key_states) -> bool:
    """The prompt (first multi-token update) attends to uncompressed K/V when the spec asks for it."""
    return spec.prompt_attention == "dense" and start == 0 and key_states.shape[-2] > 1


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
