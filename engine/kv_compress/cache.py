"""Monkeypatch Transformers Cache.update so new K/V go through compress_kv.

Llama/Qwen call update after RoPE and use the returned tensors for attention,
so this is the shared injection point. Key quantize also closes sequence groups
that filled on this update, including tokens stored by earlier steps, and writes
them back into the cache attention is about to read. Adjacent-pair pooling does
the same for keys and values when a chunk finishes a pair that an earlier step
left open. patch_cache_update returns uninstall().
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
        _close_filled_pairs(keys, values, layer_idx, spec, start, end, rope)
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


def _close_filled_pairs(keys, values, layer_idx: int, spec: KvSpec, start: int, end: int, rope) -> None:
    """Pool token pairs that this append finished. An open tail stays exact.

    A chunk that started on an even position already pooled its own full pairs.
    A chunk that started on an odd position left the previous token exact until
    here. Keys and values share the window. Each is rewritten only when this
    layer is selected for that tensor. ``pair_pool`` is full pooling with no
    RoPE alignment. ``pair_quant`` pools with a quantized residual. ``pair_gate`` merges only pairs that pass a similarity test. A shorter
    tensor has dropped its prefix.
    """
    if start % 2 == 0:
        return
    from compression_topics.spatial.algorithms.residual_pool import write_closed_pairs

    open_start = (start // 2) * 2
    closed = (end // 2) * 2
    if closed <= open_start:
        return
    for step in spec.pipeline:
        if step.method == "pair_quant":
            _close_pair_quant(step, keys, values, layer_idx, open_start, closed, end)
            continue
        if step.method == "pair_gate":
            _close_pair_gate(step, keys, values, layer_idx, open_start, closed, end)
            continue
        if step.method in {"pair_rank", "pair_rank_residual"}:
            _close_pair_rank(step, keys, values, layer_idx, open_start, closed, end)
            continue
        if step.method not in {"residual_pool", "pair_pool"}:
            continue
        if step.method == "pair_pool":
            prune_pct = 100
            align = False
        else:
            prune_pct = int(step.kwargs.get("prune_pct", 25))
            align = bool(step.kwargs.get("rope", False))
        if _enabled(step.k_layers, layer_idx) and keys.shape[-2] >= end:
            write_closed_pairs(
                keys,
                prune_pct=prune_pct,
                rope=align,
                rope_tables=rope,
                start=open_start,
                end=closed,
            )
        if _enabled(step.v_layers, layer_idx) and values.shape[-2] >= end:
            write_closed_pairs(
                values,
                prune_pct=prune_pct,
                rope=False,
                rope_tables=None,
                start=open_start,
                end=closed,
            )


def _close_pair_quant(step, keys, values, layer_idx: int, open_start: int, closed: int, end: int) -> None:
    from compression_topics.spatial.algorithms import pair_quant

    bits = int(step.kwargs.get("bits", 8))
    group = int(step.kwargs.get("group", 32))
    if _enabled(step.k_layers, layer_idx) and keys.shape[-2] >= end:
        pair_quant.write_closed_pairs(keys, bits=bits, group=group, start=open_start, end=closed)
    if _enabled(step.v_layers, layer_idx) and values.shape[-2] >= end:
        pair_quant.write_closed_pairs(values, bits=bits, group=group, start=open_start, end=closed)


def _close_pair_gate(step, keys, values, layer_idx: int, open_start: int, closed: int, end: int) -> None:
    from compression_topics.spatial.algorithms import pair_gate

    tau = float(step.kwargs.get("tau", 0.3))
    keep_pct = int(step.kwargs.get("keep_pct", 0))
    if _enabled(step.k_layers, layer_idx) and keys.shape[-2] >= end:
        pair_gate.write_closed_pairs(keys, tau=tau, keep_pct=keep_pct, target="k", start=open_start, end=closed)
    if _enabled(step.v_layers, layer_idx) and values.shape[-2] >= end:
        pair_gate.write_closed_pairs(values, tau=tau, keep_pct=keep_pct, target="v", start=open_start, end=closed)


def _close_pair_rank(step, keys, values, layer_idx: int, open_start: int, closed: int, end: int) -> None:
    from compression_topics.spatial.algorithms import pair_rank

    merge_pct = int(step.kwargs.get("merge_pct", 50))
    keep_pct = int(step.kwargs.get("keep_pct", 25 if step.method == "pair_rank_residual" else 0))
    for target, tensor, selection in (("k", keys, step.k_layers), ("v", values, step.v_layers)):
        if _enabled(selection, layer_idx) and tensor.shape[-2] >= end:
            pair_rank.write_closed_pairs(
                tensor, merge_pct=merge_pct, keep_pct=keep_pct, target=target, layer_idx=layer_idx, start=open_start, end=closed
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
