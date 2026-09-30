"""Apply a KvSpec to one layer's new K and V tensors.

Walks pipeline steps, and for each step calls the named method on K and/or V
only if that layer is in k_layers / v_layers (or "all"). Empty pipeline is identity.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import get_method
from engine.kv_compress.rope import RopeTables
from engine.kv_compress.spec import KvSpec, LayerSelection


def compress_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    spec: KvSpec,
    seq_start: int = 0,
    rope: RopeTables | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply enabled pipeline steps to this layer's new K and V chunks.

    ``seq_start`` is how many tokens this layer already stored. Key quantize
    uses it so a chunk that begins inside an open group is left exact.
    ``rope`` is forwarded to every method. Methods that do not use it ignore it.
    """
    for step in spec.pipeline:
        method = get_method(step.method)
        kwargs = dict(step.kwargs)
        kwargs["seq_start"] = seq_start
        kwargs["rope"] = rope
        if _layer_enabled(step.k_layers, layer_idx):
            key_states = method(key_states, layer_idx=layer_idx, target="k", **kwargs)
        if _layer_enabled(step.v_layers, layer_idx):
            value_states = method(value_states, layer_idx=layer_idx, target="v", **kwargs)
    return key_states, value_states


def _layer_enabled(selection: LayerSelection, layer_idx: int) -> bool:
    if selection == "all":
        return True
    return layer_idx in selection
