"""Apply a KvSpec to one layer's new K and V tensors.

Walks pipeline steps, and for each step calls the named method on K and/or V
only if that layer is in k_layers / v_layers (or "all"). Empty pipeline is identity.
"""

from __future__ import annotations

import torch

from kv_compress.methods import get_method
from kv_compress.spec import KvSpec, LayerSelection


def compress_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    spec: KvSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply enabled pipeline steps to this layer's new K and V chunks."""
    for step in spec.pipeline:
        method = get_method(step.method)
        if _layer_enabled(step.k_layers, layer_idx):
            key_states = method(key_states, layer_idx=layer_idx, target="k", **step.kwargs)
        if _layer_enabled(step.v_layers, layer_idx):
            value_states = method(value_states, layer_idx=layer_idx, target="v", **step.kwargs)
    return key_states, value_states


def _layer_enabled(selection: LayerSelection, layer_idx: int) -> bool:
    if selection == "all":
        return True
    return layer_idx in selection
