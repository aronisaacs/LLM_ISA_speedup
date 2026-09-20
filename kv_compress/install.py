from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kv_compress.cache import patch_cache_update
from kv_compress.spec import KvSpec, LayerSelection


def install(lm: Any, spec: KvSpec) -> Callable[[], None]:
    """Attach the KV pipeline for one eval run. Identity specs do not patch.

    ``lm`` is the lm-eval model wrapper (``HFLM``). Unused for identity; later
    methods can validate layer indices against the loaded decoder.
    """
    if spec.is_identity():
        return _noop
    _validate_layer_indices(lm, spec)
    return patch_cache_update(spec)


def _noop() -> None:
    return None


def _validate_layer_indices(lm: Any, spec: KvSpec) -> None:
    num_hidden_layers = _decoder_num_hidden_layers(lm)
    if num_hidden_layers is None:
        return
    for step_index, step in enumerate(spec.pipeline):
        for label, selection in (("k_layers", step.k_layers), ("v_layers", step.v_layers)):
            _check_selection(
                selection,
                num_hidden_layers,
                f"kv.pipeline[{step_index}].{label}",
            )


def _check_selection(selection: LayerSelection, num_hidden_layers: int, label: str) -> None:
    if selection == "all":
        return
    for layer_idx in selection:
        if layer_idx >= num_hidden_layers:
            raise ValueError(
                f"{label} index {layer_idx} is out of range; decoder has "
                f"{num_hidden_layers} layers (valid indices are 0..{num_hidden_layers - 1})"
            )


def _decoder_num_hidden_layers(lm: Any) -> int | None:
    model = getattr(lm, "model", None)
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is not None:
        try:
            return len(layers)
        except TypeError:
            pass

    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    for config in (text_config, model_config, getattr(lm, "_config", None)):
        if isinstance(config, dict):
            num_hidden_layers = config.get("num_hidden_layers")
        else:
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int) and not isinstance(num_hidden_layers, bool):
            return num_hidden_layers
    return None
