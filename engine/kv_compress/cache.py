"""Monkeypatch Transformers Cache.update so new K/V go through compress_kv.

Llama/Qwen call update after RoPE and use the returned tensors for attention,
so this is the shared injection point. patch_cache_update returns uninstall().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import KvSpec, LayerSelection

_original_update: Callable[..., Any] | None = None


def patch_cache_update(spec: KvSpec, rope: RopeTables | None = None) -> Callable[[], None]:
    """Wrap ``Cache.update`` so new K/V pass through ``spec`` before they are stored.

    A key step with ``pre_rope`` true is undone, compressed, then rotated back.
    Values are never rotated. Returns a zero-arg uninstall function.
    """
    from transformers.cache_utils import Cache

    global _original_update
    saved = _original_update
    if saved is None:
        saved = Cache.update
        _original_update = saved
    previous = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if _keys_want_prerope(spec, layer_idx):
            if rope is None:
                raise ValueError("pre_rope key compression requires rope_theta and head_dim")
            positions = _positions(self, layer_idx, key_states.shape[-2], key_states.device)
            cos, sin = rope.cos_sin(positions, key_states.dtype)
            key_states = apply_rope(key_states, cos, sin, inverse=True)
            key_states, value_states = compress_kv(key_states, value_states, layer_idx, spec)
            key_states = apply_rope(key_states, cos, sin, inverse=False)
        else:
            key_states, value_states = compress_kv(key_states, value_states, layer_idx, spec)
        return saved(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update = update

    def uninstall() -> None:
        Cache.update = previous

    return uninstall


def _keys_want_prerope(spec: KvSpec, layer_idx: int) -> bool:
    for step in spec.pipeline:
        if step.kwargs.get("pre_rope") and _enabled(step.k_layers, layer_idx):
            return True
    return False


def _enabled(selection: LayerSelection, layer_idx: int) -> bool:
    if selection == "all":
        return True
    return layer_idx in selection


def _positions(cache, layer_idx: int, length: int, device) -> torch.Tensor:
    start = 0
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        try:
            start = int(getter(layer_idx))
        except TypeError:
            start = int(getter())
    return torch.arange(start, start + length, device=device)
