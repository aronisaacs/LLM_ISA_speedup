"""Bitmask storage cost and net saving of a sparsified K or V vector.

A compressed vector stores its kept bf16 values plus one mask bit per value, or
one bit per RoPE pair for RoPE-paired keys. Uncompressed slots store no mask.

  net saving = (16 x values zeroed - mask bits) / (128 x 16)
"""

from __future__ import annotations

HEAD_DIM = 128
VALUE_BITS = 16
PAIR_METHODS = ("vector_compress_pair",)


def bitmask_bits(method: str, target: str, head_dim: int = HEAD_DIM) -> int:
    """Bitmask cost per compressed vector: one bit per value, or per pair for RoPE-paired keys."""
    return head_dim // 2 if method in PAIR_METHODS and target == "k" else head_dim


def zeroed_values(method: str, target: str, level: int, head_dim: int = HEAD_DIM) -> int:
    """Values set to zero at ``level`` percent, as the methods round it."""
    if method in PAIR_METHODS and target == "k":
        return 2 * ((head_dim // 2) * level // 100)
    return head_dim * level // 100


def net_saving(method: str, target: str, level: int, head_dim: int = HEAD_DIM) -> float:
    """Fraction of the vector's bits saved after the bitmask. 0 when uncompressed."""
    if level <= 0:
        return 0.0
    removed = VALUE_BITS * zeroed_values(method, target, level, head_dim)
    return (removed - bitmask_bits(method, target, head_dim)) / (head_dim * VALUE_BITS)


def net_saving_for(method: str, head_dim: int = HEAD_DIM):
    """``saving(slot, rung)`` for the greedy: net of the bitmask."""
    return lambda slot, rung: net_saving(method, slot.target, rung.level, head_dim)
