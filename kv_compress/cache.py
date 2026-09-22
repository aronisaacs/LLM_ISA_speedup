"""Monkeypatch Transformers Cache.update so new K/V go through compress_kv.

Llama/Qwen call update after RoPE and use the returned tensors for attention,
so this is the shared injection point. patch_cache_update returns uninstall().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kv_compress.pipeline import compress_kv
from kv_compress.spec import KvSpec

_original_update: Callable[..., Any] | None = None


def patch_cache_update(spec: KvSpec) -> Callable[[], None]:
    """Wrap ``Cache.update`` so new K/V pass through ``spec`` before they are stored.

    Returns a zero-arg uninstall function that restores the previous ``update``.
    """
    from transformers.cache_utils import Cache

    global _original_update
    saved = _original_update
    if saved is None:
        saved = Cache.update
        _original_update = saved
    previous = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        key_states, value_states = compress_kv(key_states, value_states, layer_idx, spec)
        return saved(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update = update

    def uninstall() -> None:
        Cache.update = previous

    return uninstall
