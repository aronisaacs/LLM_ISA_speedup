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


def keys_only(n_layers: int) -> tuple[Slot, ...]:
    _check_n_layers(n_layers)
    return tuple(Slot(index, "k") for index in range(n_layers))


def mix(n_layers: int) -> tuple[Slot, ...]:
    _check_n_layers(n_layers)
    slots = []
    for index in range(n_layers):
        slots.append(Slot(index, "k"))
        slots.append(Slot(index, "v"))
    return tuple(slots)


def space_slots(space: str, n_layers: int) -> tuple[Slot, ...]:
    if space == "keys_only":
        return keys_only(n_layers)
    if space == "mix":
        return mix(n_layers)
    raise ValueError("space must be 'keys_only' or 'mix'")


def n_pick(n_slots: int, budget: float) -> int:
    """How many slots a budget in [0, 1] turns on (floor, except budget 1 → all)."""
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise TypeError("budget must be a real number in [0, 1]")
    if budget < 0 or budget > 1:
        raise ValueError("budget must be in [0, 1]")
    if n_slots < 0:
        raise ValueError("n_slots must be non-negative")
    if budget == 1:
        return n_slots
    return int(n_slots * budget)


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
