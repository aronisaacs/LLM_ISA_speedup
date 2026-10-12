"""Attach or skip KV compression for one configuration.

install(lm, spec) is a no-op for an empty pipeline (dense baseline). Otherwise
it checks layer indices against the decoder and patches Cache.update. Always
call the returned uninstall() so the next configuration can use a different spec.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.kv_compress.cache import patch_cache_update
from engine.kv_compress.rope import rope_from_config
from engine.kv_compress.spec import KvSpec, LayerSelection


def install(lm: Any, spec: KvSpec) -> Callable[[], None]:
    """Attach the KV pipeline for one configuration. Identity specs do not patch.

    ``lm`` is the lm-eval model wrapper (``HFLM``). Unused for identity; later
    methods can validate layer indices against the loaded decoder.
    """
    if spec.is_identity():
        return _noop
    _validate_layer_indices(lm, spec)
    return patch_cache_update(spec, _rope_tables(lm))


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


def _rope_tables(lm: Any):
    tables = rope_from_config(_decoder_config(lm))
    if tables is None:
        return None
    decoder = _decoder_module(lm)
    rotary = getattr(decoder, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", 1.0)
    if not isinstance(scaling, (int, float)) or isinstance(scaling, bool):
        scaling = 1.0
    if inv_freq is None and scaling == 1.0:
        return tables
    return type(tables)(
        rope_theta=tables.rope_theta,
        head_dim=tables.head_dim,
        inv_freq=None if inv_freq is None else tuple(float(value) for value in inv_freq.detach().float().cpu().tolist()),
        attention_scaling=float(scaling),
    )


def _decoder_module(lm: Any):
    model = getattr(lm, "model", None)
    return getattr(model, "model", model)


def _decoder_config(lm: Any):
    model = getattr(lm, "model", None)
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    return text_config or model_config or getattr(lm, "_config", None)


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
