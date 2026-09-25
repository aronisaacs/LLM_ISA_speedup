"""Parse and validate the JSON ``kv`` object on a configuration.

Turns {pipeline: [{method, k_layers, v_layers, ...}]} into KvSpec / PipelineStep.
Unknown methods or keys fail here so a typo cannot silently run dense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from engine.kv_compress.methods import METHODS

LayerSelection = Literal["all"] | frozenset[int]

_ALLOWED_KV_KEYS = {"pipeline"}
_STEP_RESERVED_KEYS = {"method", "k_layers", "v_layers"}


@dataclass(frozen=True)
class PipelineStep:
    method: str
    k_layers: LayerSelection
    v_layers: LayerSelection
    kwargs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "k_layers": _layers_to_json(self.k_layers),
            "v_layers": _layers_to_json(self.v_layers),
        }
        payload.update(self.kwargs)
        return payload


@dataclass(frozen=True)
class KvSpec:
    pipeline: tuple[PipelineStep, ...] = ()

    def is_identity(self) -> bool:
        return len(self.pipeline) == 0

    def to_dict(self) -> dict[str, Any]:
        return {"pipeline": [step.to_dict() for step in self.pipeline]}


def parse_kv_spec(value: Any | None) -> KvSpec:
    """Parse a JSON ``kv`` object. ``None`` or ``{}`` means dense (identity)."""
    if value is None:
        return KvSpec()
    if not isinstance(value, dict):
        raise TypeError("kv must be a mapping or omitted")
    unknown = set(value) - _ALLOWED_KV_KEYS
    if unknown:
        raise ValueError(f"kv contains unknown keys: {sorted(unknown, key=str)!r}")
    pipeline_value = value.get("pipeline", [])
    if pipeline_value is None:
        pipeline_value = []
    if not isinstance(pipeline_value, list):
        raise TypeError("kv.pipeline must be a list")
    return KvSpec(pipeline=tuple(_parse_step(step, index) for index, step in enumerate(pipeline_value)))


def _parse_step(step: Any, index: int) -> PipelineStep:
    if not isinstance(step, dict):
        raise TypeError(f"kv.pipeline[{index}] must be a mapping")
    method = step.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"kv.pipeline[{index}].method must be a non-empty string")
    if method not in METHODS:
        known = ", ".join(sorted(METHODS)) or "(none registered)"
        raise ValueError(
            f"kv.pipeline[{index}] uses unknown method {method!r}. Known methods: {known}"
        )
    kwargs = {key: copy_value for key, copy_value in step.items() if key not in _STEP_RESERVED_KEYS}
    return PipelineStep(
        method=method,
        k_layers=_parse_layers(step.get("k_layers"), f"kv.pipeline[{index}].k_layers"),
        v_layers=_parse_layers(step.get("v_layers"), f"kv.pipeline[{index}].v_layers"),
        kwargs=dict(kwargs),
    )


def _parse_layers(value: Any, label: str) -> LayerSelection:
    if value is None:
        return frozenset()
    if value == "all":
        return "all"
    if isinstance(value, bool) or not isinstance(value, list):
        raise TypeError(f"{label} must be 'all', a list of layer indices, or omitted")
    layers: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{label} entries must be integers")
        if item < 0:
            raise ValueError(f"{label} entries must be non-negative")
        layers.add(item)
    return frozenset(layers)


def _layers_to_json(layers: LayerSelection) -> str | list[int]:
    if layers == "all":
        return "all"
    return sorted(layers)
