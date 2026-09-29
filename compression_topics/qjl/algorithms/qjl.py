"""Hadamard codebook stand-in. Stock attention still runs on full-width tensors.

Each token is rotated by a fixed diagonal of ±1 and a Walsh–Hadamard transform,
snapped onto a Lloyd–Max grid for a unit-vector coordinate, then rotated back
and rescaled by an fp16 norm. Keys and values share this path. Post-RoPE.
"""

from __future__ import annotations

import math

import torch

from engine.kv_compress.methods import METHODS

_SIGNS: dict[tuple[int, int], torch.Tensor] = {}
_STANDARD_CENTROIDS: dict[int, torch.Tensor] = {}


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    bits: int = 4,
    **_unused,
) -> torch.Tensor:
    del _unused
    _check_bits(bits)
    if target not in {"k", "v"}:
        raise ValueError(f"qjl target must be 'k' or 'v', got {target!r}")
    return _quantize(tensor, layer_idx, bits)


def sign_diagonal(layer_idx: int, features: int) -> torch.Tensor:
    """Fixed ±1 diagonal for one layer, shape ``[features]``."""
    key = (int(layer_idx), int(features))
    cached = _SIGNS.get(key)
    if cached is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10_000 + int(layer_idx))
        cached = torch.randint(0, 2, (features,), generator=generator, dtype=torch.int8)
        cached = cached.mul(2).sub(1).to(dtype=torch.float32)
        _SIGNS[key] = cached
    return cached


def hadamard(tensor: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh–Hadamard along the last dimension."""
    features = tensor.shape[-1]
    if features < 1 or features & (features - 1):
        raise ValueError(f"hadamard length must be a power of two, got {features}")
    values = tensor.float()
    step = 1
    while step < features:
        values = values.reshape(*values.shape[:-1], features // (2 * step), 2, step)
        left = values[..., 0, :]
        right = values[..., 1, :]
        values = torch.stack((left + right, left - right), dim=-2).reshape(*tensor.shape[:-1], features)
        step *= 2
    return values / math.sqrt(features)


def centroids(bits: int, features: int) -> torch.Tensor:
    """Lloyd–Max levels for one coordinate of a unit vector in ``features`` dimensions."""
    _check_bits(bits)
    standard = _STANDARD_CENTROIDS.get(bits)
    if standard is None:
        standard = _lloyd_max(1 << bits)
        _STANDARD_CENTROIDS[bits] = standard
    return standard / math.sqrt(features)


def _quantize(tensor: torch.Tensor, layer_idx: int, bits: int) -> torch.Tensor:
    features = tensor.shape[-1]
    flat = tensor.reshape(-1, features).float()
    norm = flat.norm(dim=-1, keepdim=True)
    stored = norm.to(dtype=torch.float16).float()
    unit = torch.where(norm == 0, torch.zeros_like(flat), flat / norm.clamp_min(torch.finfo(torch.float32).tiny))
    signs = sign_diagonal(layer_idx, features).to(device=flat.device)
    rotated = hadamard(unit * signs)
    levels = centroids(bits, features).to(device=flat.device)
    index = (rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    restored = hadamard(levels[index]) * signs
    rebuilt = torch.where(norm == 0, torch.zeros_like(restored), restored * stored)
    return rebuilt.reshape_as(tensor).to(dtype=tensor.dtype)


def _lloyd_max(levels: int) -> torch.Tensor:
    centers = [_inverse_cdf((index + 0.5) / levels) for index in range(levels)]
    for _ in range(40):
        bounds = [math.inf * -1.0]
        bounds.extend(0.5 * (centers[index] + centers[index + 1]) for index in range(levels - 1))
        bounds.append(math.inf)
        updated = []
        for index in range(levels):
            left, right = bounds[index], bounds[index + 1]
            mass = _cdf(right) - _cdf(left)
            updated.append((_pdf(left) - _pdf(right)) / mass)
        if max(abs(new - old) for new, old in zip(updated, centers)) < 1e-12:
            centers = updated
            break
        centers = updated
    return torch.tensor(centers, dtype=torch.float32)


def _pdf(point: float) -> float:
    if math.isinf(point):
        return 0.0
    return math.exp(-0.5 * point * point) / math.sqrt(2.0 * math.pi)


def _cdf(point: float) -> float:
    if point == math.inf:
        return 1.0
    if point == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf(point / math.sqrt(2.0)))


def _inverse_cdf(probability: float) -> float:
    return math.sqrt(2.0) * _inverse_erf(2.0 * probability - 1.0)


def _inverse_erf(value: float) -> float:
    sign = 1.0 if value >= 0.0 else -1.0
    log_term = math.log(max(1.0 - value * value, 1e-300))
    first = 2.0 / (math.pi * 0.147) + 0.5 * log_term
    return sign * math.sqrt(math.sqrt(first * first - log_term / 0.147) - first)


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1 or bits > 8:
        raise ValueError("qjl bits must be an integer 1..8")


METHODS["qjl"] = apply
