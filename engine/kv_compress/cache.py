"""Monkeypatch Transformers Cache.update so new K/V go through compress_kv.

Llama/Qwen call update after RoPE and use the returned tensors for attention,
so this is the shared injection point. Key quantize also closes sequence groups
that filled on this update, including tokens stored by earlier steps, and writes
them back into the cache attention is about to read. patch_cache_update returns
uninstall().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compression_topics.quantize.algorithms.quantize import write_closed_key_groups
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables
from engine.kv_compress.spec import KvSpec, LayerSelection, PipelineStep

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

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        start = _seq_length(self, layer_idx)
        end = start + key_states.shape[-2]
        key_states, value_states = compress_kv(
            key_states, value_states, layer_idx, spec, seq_start=start, rope=rope
        )
        keys, values = saved(self, key_states, value_states, layer_idx, *args, **kwargs)
        _close_filled_key_groups(keys, layer_idx, spec, start, end)
        return keys, values

    Cache.update = update

    def uninstall() -> None:
        Cache.update = previous

    return uninstall


def _close_filled_key_groups(keys, layer_idx: int, spec: KvSpec, start: int, end: int) -> None:
    """Quantize key groups that this append finished. An open tail stays exact.

    Groups that began on a group boundary were already quantized with the chunk.
    A group that started on an earlier step is still full precision until here.
    ``keys`` is the tensor attention will read. Dynamic and static caches return
    their own storage, so the write updates the cache too. A shorter tensor has
    dropped its prefix, and absolute positions would land on the wrong tokens.
    """
    step = _key_quantize_step(spec, layer_idx)
    if step is None or keys.shape[-2] < end:
        return
    group = int(step.kwargs.get("group", 32))
    if group < 1 or start % group == 0:
        return
    open_start = (start // group) * group
    closed = (end // group) * group
    if closed <= open_start:
        return
    write_closed_key_groups(
        keys,
        bits=int(step.kwargs.get("bits", 8)),
        group=group,
        start=open_start,
        end=closed,
    )


def _key_quantize_step(spec: KvSpec, layer_idx: int) -> PipelineStep | None:
    for step in spec.pipeline:
        if step.method == "quantize" and _enabled(step.k_layers, layer_idx):
            return step
    return None


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
