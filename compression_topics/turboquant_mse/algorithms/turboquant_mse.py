"""TurboQuant-MSE (no QJL residual). Stock attention still runs on full-width tensors.

Each token is rotated by a fixed diagonal of ±1 signs (D) and a Walsh–Hadamard
transform (H), snapped onto a Lloyd–Max grid for a unit-vector coordinate (Gaussian
approximation scaled by 1/sqrt(d)), then rotated back and rescaled by an fp16 norm.
Keys and values share this path. Post-RoPE, in ``Cache.update``, so prefill attention
also sees the compressed keys.

Signs (``signs``):
  "rope_pairs"   each RoPE pair (i, i + d/2, the HF rotate_half pairing) shares one sign,
                 so D commutes with RoPE and can be folded into W_q and W_k offline.
                 Applied to the original channels, before any outlier split.
  "independent"  one sign per channel (runs whose kv has no ``signs`` key; earlier results).

Outlier split (``outliers`` > 0), as in the paper's 2.5- and 3.5-bit KV runs: per
layer, KV head and target, a fixed set of ``outliers`` channels (from a calibration
table, see scripts/calibrate_outliers.py) and the remaining channels are quantized
as two independent TurboQuant vectors, each with its own fp16 norm, rotation and
codebook. Outlier channels get ``outlier_bits`` (default ``bits`` + 1). With 64 of 128
channels (2 bits + 64 outliers at 3 = 2.5 bits), both parts use a 64-point Hadamard.
A part whose length is not a power of two uses a seeded random orthogonal rotation
(QR of a Gaussian matrix) instead.

The hardware order of operations (rotate the query once, sign/size sums, values'
rotations folded into W_v and W_o) gives the same scores and outputs up to rounding,
so it is not simulated separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS

ROOT = Path(__file__).resolve().parents[3]

_SIGNS: dict[tuple[int, int], torch.Tensor] = {}
_ORTHOGONAL: dict[tuple[int, int], torch.Tensor] = {}
_STANDARD_CENTROIDS: dict[int, torch.Tensor] = {}
_TABLES: dict[str, dict] = {}
# Small fixed tensors already on the model's device, and the outlier index lists per layer.
# Building them on the CPU and copying them for every layer and every generated token
# made the GPU wait on each copy; the values are the same either way.
_ON_DEVICE: dict[tuple, torch.Tensor] = {}
_SPLITS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _on_device(key: tuple, build, device: torch.device) -> torch.Tensor:
    full = (*key, str(device))
    cached = _ON_DEVICE.get(full)
    if cached is None:
        cached = build().to(device=device)
        _ON_DEVICE[full] = cached
    return cached


def apply(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    bits: int = 4,
    outliers: int = 0,
    outlier_bits: int | None = None,
    outlier_table: str | None = None,
    signs: str = "independent",
    **_unused,
) -> torch.Tensor:
    del _unused
    _check_bits(bits)
    _check_signs(signs)
    if target not in {"k", "v"}:
        raise ValueError(f"turboquant_mse target must be 'k' or 'v', got {target!r}")
    features = tensor.shape[-1]
    device = tensor.device
    if not outliers:
        vector = _on_device(("signs", signs, layer_idx, features), lambda: _sign_vector(signs, layer_idx, features), device)
        return _quantize(tensor, layer_idx, bits, signs=vector)
    high = bits + 1 if outlier_bits is None else outlier_bits
    _check_bits(high)
    split = _split_indices(outlier_table, layer_idx, target, outliers, features, device)
    if signs == "independent":
        return _quantize_split(tensor, layer_idx, bits, high, split, part_signs=True)
    # D on the original channels (so it can be folded into W_q / W_k), then each part is rotated by H alone.
    diagonal = _on_device(("signs", "rope_pairs", layer_idx, features), lambda: rope_pair_signs(layer_idx, features), device)
    flipped = (tensor.float() * diagonal).to(dtype=tensor.dtype)
    rebuilt = _quantize_split(flipped, layer_idx, bits, high, split, part_signs=False)
    return (rebuilt.float() * diagonal).to(dtype=tensor.dtype)


def validate(options: dict) -> None:
    """Before model load: a known sign scheme, and an outlier split needs a readable table with that many channels."""
    _check_signs(options.get("signs", "independent"))
    outliers = options.get("outliers", 0)
    if not outliers:
        return
    if not isinstance(outliers, int) or isinstance(outliers, bool) or outliers < 1:
        raise ValueError("turboquant_mse outliers must be a positive integer")
    if not options.get("outlier_table"):
        raise ValueError("turboquant_mse outliers needs outlier_table")
    table = load_outlier_table(options["outlier_table"])
    if table["ranking_length"] <= outliers:
        raise ValueError(f"outlier table ranks {table['ranking_length']} channels per head, asked for {outliers} outliers")


def load_outlier_table(path: str) -> dict:
    """Calibration table written by scripts/calibrate_outliers.py. Relative paths start at the repo root."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    key = str(resolved)
    table = _TABLES.get(key)
    if table is None:
        table = json.loads(resolved.read_text())
        _TABLES[key] = table
    return table


def outlier_channels(path: str | None, layer_idx: int, target: str, outliers: int) -> torch.Tensor:
    """Channel indices per KV head, shape ``[heads, outliers]``: the first ``outliers`` of the table's ranking."""
    if path is None:
        raise ValueError("turboquant_mse outliers needs outlier_table")
    ranked = load_outlier_table(path)["channels"][target][int(layer_idx)]
    return torch.tensor([head[:outliers] for head in ranked], dtype=torch.long)


def _split_indices(path: str | None, layer_idx: int, target: str, outliers: int, features: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Outlier and remaining channel indices per KV head, each ``[heads, n]`` in ascending order, on ``device``."""
    key = (path, int(layer_idx), target, int(outliers), int(features), str(device))
    cached = _SPLITS.get(key)
    if cached is None:
        channels = outlier_channels(path, layer_idx, target, outliers)
        heads = channels.shape[0]
        mask = torch.zeros(heads, features, dtype=torch.bool)
        mask.scatter_(1, channels, True)
        if int(mask.sum()) != channels.numel():
            raise ValueError("outlier channels repeat within a head")
        high_idx = torch.stack([torch.nonzero(row).squeeze(1) for row in mask])
        low_idx = torch.stack([torch.nonzero(~row).squeeze(1) for row in mask])
        cached = (high_idx.to(device=device), low_idx.to(device=device))
        _SPLITS[key] = cached
    return cached


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


def rope_pair_signs(layer_idx: int, features: int) -> torch.Tensor:
    """±1 diagonal with sign[i] == sign[i + features/2] (HF RoPE pairs), shape ``[features]``."""
    if features % 2:
        raise ValueError(f"rope_pairs signs need an even length, got {features}")
    half = sign_diagonal(layer_idx, features // 2)
    return torch.cat((half, half))


def _sign_vector(signs: str, layer_idx: int, features: int) -> torch.Tensor:
    return rope_pair_signs(layer_idx, features) if signs == "rope_pairs" else sign_diagonal(layer_idx, features)


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


def orthogonal(layer_idx: int, features: int) -> torch.Tensor:
    """Seeded random orthogonal matrix (QR of a Gaussian matrix, sign-fixed), shape ``[features, features]``."""
    key = (int(layer_idx), int(features))
    cached = _ORTHOGONAL.get(key)
    if cached is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20_000 + int(layer_idx))
        gaussian = torch.randn(features, features, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(gaussian)
        cached = (q * torch.sign(torch.diagonal(r))).to(dtype=torch.float32)
        _ORTHOGONAL[key] = cached
    return cached


def _quantize_part(tensor: torch.Tensor, layer_idx: int, bits: int, part_signs: bool) -> torch.Tensor:
    """TurboQuant on vectors of any length: Hadamard path for powers of two, random orthogonal otherwise.

    ``part_signs``: the Hadamard path draws its own ±1 diagonal for this part ("independent" signs);
    otherwise the caller has already applied D to the whole vector and the part is rotated by H alone.
    """
    features = tensor.shape[-1]
    device = tensor.device
    if not features & (features - 1):
        if part_signs:
            vector = _on_device(("signs", "independent", layer_idx, features), lambda: sign_diagonal(layer_idx, features), device)
        else:
            vector = _on_device(("ones", features), lambda: torch.ones(features), device)
        return _quantize(tensor, layer_idx, bits, signs=vector)
    flat = tensor.reshape(-1, features).float()
    norm = flat.norm(dim=-1, keepdim=True)
    stored = norm.to(dtype=torch.float16).float()
    unit = torch.where(norm == 0, torch.zeros_like(flat), flat / norm.clamp_min(torch.finfo(torch.float32).tiny))
    rotation = _on_device(("orthogonal", layer_idx, features), lambda: orthogonal(layer_idx, features), device)
    rotated = unit @ rotation.T
    levels = _on_device(("levels", bits, features), lambda: centroids(bits, features), device)
    index = (rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    restored = levels[index] @ rotation
    rebuilt = torch.where(norm == 0, torch.zeros_like(restored), restored * stored)
    return rebuilt.reshape_as(tensor)


def _quantize_split(tensor: torch.Tensor, layer_idx: int, bits: int, high: int,
                    split: tuple[torch.Tensor, torch.Tensor], part_signs: bool) -> torch.Tensor:
    """Outlier channels at ``high`` bits and the rest at ``bits``, two independent TurboQuant vectors per head.

    ``tensor`` is ``[batch, kv_heads, seq, head_dim]``; ``split`` is the (outlier, remaining) channel
    indices from ``_split_indices``, each ``[kv_heads, n]``.
    """
    batch, heads, seq, features = tensor.shape
    high_idx, low_idx = split
    if high_idx.shape[0] != heads:
        raise ValueError(f"outlier table has {high_idx.shape[0]} heads, tensor has {heads}")
    values = tensor.float()
    rebuilt = torch.empty_like(values)
    for index, part_bits in ((high_idx, high), (low_idx, bits)):
        expanded = index.view(1, heads, 1, -1).expand(batch, heads, seq, -1)
        part = torch.gather(values, -1, expanded)
        rebuilt.scatter_(-1, expanded, _quantize_part(part, layer_idx, part_bits, part_signs))
    return rebuilt.to(dtype=tensor.dtype)


def _quantize(tensor: torch.Tensor, layer_idx: int, bits: int, signs: torch.Tensor) -> torch.Tensor:
    features = tensor.shape[-1]
    flat = tensor.reshape(-1, features).float()
    norm = flat.norm(dim=-1, keepdim=True)
    stored = norm.to(dtype=torch.float16).float()
    unit = torch.where(norm == 0, torch.zeros_like(flat), flat / norm.clamp_min(torch.finfo(torch.float32).tiny))
    signs = signs.to(device=flat.device)
    rotated = hadamard(unit * signs)
    levels = _on_device(("levels", bits, features), lambda: centroids(bits, features), flat.device)
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


def _check_signs(signs) -> None:
    if signs not in {"independent", "rope_pairs"}:
        raise ValueError(f"turboquant_mse signs must be 'independent' or 'rope_pairs', got {signs!r}")


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1 or bits > 8:
        raise ValueError("turboquant_mse bits must be an integer 1..8")


METHODS["turboquant_mse"] = apply
OPTION_VALIDATORS["turboquant_mse"] = validate
