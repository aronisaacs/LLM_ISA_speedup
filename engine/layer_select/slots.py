"""A slot is one K or V tensor at one layer (the unit greedy turns on)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Target = Literal["k", "v"]


@dataclass(frozen=True, order=True)
class Slot:
    layer: int
    target: Target

    def tag(self) -> str:
        return f"{self.target}{self.layer:02d}"

    def to_dict(self) -> dict:
        return {"layer": self.layer, "target": self.target}

    @classmethod
    def from_dict(cls, payload: dict) -> "Slot":
        return cls(int(payload["layer"]), payload["target"])


def all_slots(n_layers: int) -> tuple[Slot, ...]:
    """Every key and every value, one slot per tensor."""
    _check_n_layers(n_layers)
    slots = []
    for index in range(n_layers):
        slots.append(Slot(index, "k"))
        slots.append(Slot(index, "v"))
    return tuple(slots)


def pretrained_from_model_args(model_args) -> str:
    if isinstance(model_args, dict):
        value = model_args.get("pretrained")
        if not value:
            raise ValueError("model_args dict has no pretrained")
        return str(value)
    for part in str(model_args).split(","):
        key, _, value = part.partition("=")
        if key.strip() == "pretrained" and value.strip():
            return value.strip()
    raise ValueError("model_args has no pretrained=")


def n_hidden_layers(model_args) -> int:
    """Layer count from Hugging Face config (no weights)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(pretrained_from_model_args(model_args))
    return int(config.num_hidden_layers)


def _check_n_layers(n_layers: int) -> None:
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")
