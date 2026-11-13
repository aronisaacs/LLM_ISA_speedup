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

Implementation: per layer and target, D, the outlier split and the part rotations are folded
into one rotation matrix per KV head (built once from the pieces above), so each call is a
batched matmul, a nearest-level lookup per part and a matmul back: ~15-25 GPU operations
instead of ~125-240, which matters for every generated token.

Paper-faithful options (for one check against the paper's Table 1; not our design):
  ``rotation="orthogonal"``  every part uses a seeded random orthogonal matrix (QR of a Gaussian
                             matrix, the paper's Section 3.1) instead of a Hadamard transform.
  ``codebook="beta"``        Lloyd–Max levels for the exact distribution of one coordinate of a
                             random unit vector in d dimensions, density ∝ (1 - x²)^((d-3)/2),
                             instead of the Gaussian approximation N(0, 1/d).

  ``outlier_group=G``         dynamic outliers, as in the QJL code by the paper's first author: for
                             each aligned group of G tokens, per KV head, the ``outliers`` channels
                             with the largest norm over the group (instead of the calibration table).
                             Needs a recent window (the cache ages tokens out in whole groups). The
                             channel indices this needs are not counted in the bits per value.

``recent_window`` (tokens) is read by the cache (engine/kv_compress/cache.py): the newest
``recent_window`` tokens stay exact and each token is compressed when it leaves the window.
This method compresses every vector on its own, so that equals compressing on arrival and
reading the window's exact copies.

The hardware order of operations (rotate the query once, sign/size sums, values'
rotations folded into W_v and W_o) gives the same scores and outputs up to rounding,
so it is not simulated separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS, WINDOW_GROUPS

ROOT = Path(__file__).resolve().parents[3]

_SIGNS: dict[tuple[int, int], torch.Tensor] = {}
_ORTHOGONAL: dict[tuple[int, int], torch.Tensor] = {}
_STANDARD_CENTROIDS: dict[int, torch.Tensor] = {}
_BETA_CENTROIDS: dict[tuple[int, int], torch.Tensor] = {}
ROTATIONS = ("hadamard", "orthogonal")
CODEBOOKS = ("gaussian", "beta")
_TABLES: dict[str, dict] = {}
# One plan per (layer, target, options, device): the per-head rotation matrix and its segments.
_PLANS: dict[tuple, "_Plan"] = {}


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
    rotation: str = "hadamard",
    codebook: str = "gaussian",
    recent_window: int = 0,
    outlier_group: int = 0,
    **_unused,
) -> torch.Tensor:
    del _unused, recent_window  # the window is the cache's job
    _check_bits(bits)
    _check_signs(signs)
    _check_choice("rotation", rotation, ROTATIONS)
    _check_choice("codebook", codebook, CODEBOOKS)
    if target not in {"k", "v"}:
        raise ValueError(f"turboquant_mse target must be 'k' or 'v', got {target!r}")
    high = bits + 1 if outlier_bits is None else outlier_bits
    if outliers:
        _check_bits(high)
    if outlier_group:
        return _quantize_grouped(tensor, layer_idx, bits, outliers, high, signs, outlier_group, rotation, codebook)
    plan = _plan(layer_idx, target, bits, outliers, high, outlier_table, signs, tensor.shape[1], tensor.shape[-1],
                 tensor.device, rotation, codebook)
    return _quantize_with(plan, tensor)


def validate(options: dict) -> None:
    """Before model load: a known sign scheme, and an outlier split needs a readable table with that many channels."""
    _check_signs(options.get("signs", "independent"))
    _check_choice("rotation", options.get("rotation", "hadamard"), ROTATIONS)
    _check_choice("codebook", options.get("codebook", "gaussian"), CODEBOOKS)
    window = options.get("recent_window", 0)
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("turboquant_mse recent_window must be a non-negative integer")
    outliers = options.get("outliers", 0)
    group = options.get("outlier_group", 0)
    if not isinstance(group, int) or isinstance(group, bool) or group < 0:
        raise ValueError("turboquant_mse outlier_group must be a non-negative integer")
    if group and not (outliers and window):
        raise ValueError("turboquant_mse outlier_group needs outliers and a recent_window")
    if not outliers:
        return
    if not isinstance(outliers, int) or isinstance(outliers, bool) or outliers < 1:
        raise ValueError("turboquant_mse outliers must be a positive integer")
    if group:
        return
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


def beta_centroids(bits: int, features: int) -> torch.Tensor:
    """Lloyd–Max levels for the exact density of one coordinate of a random unit vector in ``features``
    dimensions, f(x) ∝ (1 - x²)^((features - 3) / 2) on [-1, 1] (the paper's Lemma 1), on a fine grid."""
    _check_bits(bits)
    key = (int(bits), int(features))
    cached = _BETA_CENTROIDS.get(key)
    if cached is None:
        grid = torch.linspace(-1, 1, 400_001, dtype=torch.float64)[1:-1]
        weight = (1 - grid * grid).pow((features - 3) / 2)
        weight = weight / weight.sum()
        centers = centroids(bits, features).double()
        for _ in range(500):
            bounds = (centers[1:] + centers[:-1]) / 2
            cell = torch.bucketize(grid, bounds)
            mass = torch.zeros_like(centers).index_add_(0, cell, weight)
            first = torch.zeros_like(centers).index_add_(0, cell, weight * grid)
            updated = first / mass
            updated = (updated - updated.flip(0)) / 2  # the density is symmetric; keep the levels exactly so
            if (updated - centers).abs().max() < 1e-12:
                centers = updated
                break
            centers = updated
        cached = centers.to(torch.float32)
        _BETA_CENTROIDS[key] = cached
    return cached


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


class _Plan:
    """Per-head rotation ``matrix`` [heads or 1, d, d] (rows in, columns out: r = x @ matrix) and the
    segments of r that are quantized as separate TurboQuant vectors: (start, length, levels, boundaries)."""

    def __init__(self, matrix: torch.Tensor, segments: list[tuple[int, int, torch.Tensor, torch.Tensor]]):
        self.matrix = matrix
        self.back = matrix.transpose(-1, -2).contiguous()
        self.segments = segments


def _plan(layer_idx: int, target: str, bits: int, outliers: int, high: int, table: str | None, signs: str,
          heads: int, features: int, device: torch.device, rotation: str = "hadamard",
          codebook: str = "gaussian") -> _Plan:
    """Build the transform once from the reference pieces (signs, Hadamard, random orthogonal, outlier split).

    The rotation is pushed through the identity, so the matrix is exactly the linear map those pieces define:
    D on the original channels, then each part (all channels, or outliers and the rest) rotated on its own.
    """
    key = (int(layer_idx), target, int(bits), int(outliers), int(high), table, signs, int(heads), int(features),
           str(device), rotation, codebook)
    cached = _PLANS.get(key)
    if cached is not None:
        return cached
    eye = torch.eye(features, dtype=torch.float64)
    if not outliers:
        parts = [(torch.arange(features), bits)]
    else:
        channels = outlier_channels(table, layer_idx, target, outliers)
        if channels.shape[0] != heads:
            raise ValueError(f"outlier table has {channels.shape[0]} heads, tensor has {heads}")
        mask = torch.zeros(heads, features, dtype=torch.bool)
        mask.scatter_(1, channels, True)
        if int(mask.sum()) != channels.numel():
            raise ValueError("outlier channels repeat within a head")
        parts = [(torch.stack([torch.nonzero(row).squeeze(1) for row in mask]), high),
                 (torch.stack([torch.nonzero(~row).squeeze(1) for row in mask]), bits)]
    whole_signs = signs == "rope_pairs" or not outliers
    flipped = eye * _sign_vector(signs, layer_idx, features).double() if whole_signs else eye
    per_head = []
    for head in range(heads if outliers else 1):
        columns = []
        for index, _ in parts:
            chosen = index if index.dim() == 1 else index[head]
            columns.append(_rotate_part(flipped[:, chosen], layer_idx, part_signs=not whole_signs,
                                        dense=rotation == "orthogonal"))
        per_head.append(torch.cat(columns, dim=1))
    matrix = torch.stack(per_head).to(dtype=torch.float32, device=device)
    segments, start = [], 0
    for index, part_bits in parts:
        length = index.shape[-1]
        levels = beta_centroids(part_bits, length) if codebook == "beta" else centroids(part_bits, length)
        boundaries = (levels[1:] + levels[:-1]) / 2
        segments.append((start, length, levels.to(device), boundaries.to(device)))
        start += length
    cached = _Plan(matrix, segments)
    _PLANS[key] = cached
    return cached


def _rotate_part(rows: torch.Tensor, layer_idx: int, part_signs: bool, dense: bool = False) -> torch.Tensor:
    """The reference rotation of one part, applied to each row: Hadamard (with its own signs when
    ``part_signs``) for power-of-two lengths, the seeded random orthogonal matrix otherwise or when ``dense``."""
    length = rows.shape[-1]
    if not dense and not length & (length - 1):
        if part_signs:
            rows = rows * sign_diagonal(layer_idx, length).double()
        return _hadamard64(rows)
    return rows @ orthogonal(layer_idx, length).double().T


def _hadamard64(rows: torch.Tensor) -> torch.Tensor:
    """``hadamard`` in float64 (it computes in float32), for building the matrix exactly."""
    features = rows.shape[-1]
    values = rows.double()
    step = 1
    while step < features:
        values = values.reshape(*values.shape[:-1], features // (2 * step), 2, step)
        left, right = values[..., 0, :], values[..., 1, :]
        values = torch.stack((left + right, left - right), dim=-2).reshape(*rows.shape[:-1], features)
        step *= 2
    return values / math.sqrt(features)


def _quantize_grouped(tensor: torch.Tensor, layer_idx: int, bits: int, outliers: int, high: int, signs: str,
                      group: int, rotation: str, codebook: str) -> torch.Tensor:
    """Dynamic outliers: per aligned group of ``group`` tokens and KV head, the ``outliers`` channels with the
    largest norm over the group get ``high`` bits; each part is rotated (fixed per part length) and quantized
    against its own norm and codebook, as in the fixed split. Tokens must come in whole groups."""
    batch, heads, tokens, features = tensor.shape
    if tokens % group:
        raise ValueError(f"turboquant_mse outlier_group={group} needs whole groups, got {tokens} tokens")
    values = tensor.float().reshape(batch, heads, tokens // group, group, features)
    if signs == "rope_pairs":
        values = values * rope_pair_signs(layer_idx, features).to(values.device)
    chosen = values.norm(dim=-2).topk(outliers, dim=-1).indices.sort(dim=-1).values  # [b, h, groups, outliers]
    mask = torch.zeros(batch, heads, tokens // group, features, dtype=torch.bool, device=values.device)
    mask.scatter_(-1, chosen, True)
    rest = torch.nonzero(~mask.reshape(-1, features))[:, 1].reshape(*mask.shape[:-1], features - outliers)
    restored = torch.empty_like(values)
    with _full_fp32_matmul(values.device):
        for index, part_bits in ((chosen, high), (rest, bits)):
            length = index.shape[-1]
            gather = index.unsqueeze(-2).expand(-1, -1, -1, group, -1)
            part = values.gather(-1, gather)
            matrix = _part_matrix(layer_idx, length, signs, rotation).to(values.device)
            rotated = part @ matrix
            norm = rotated.norm(dim=-1, keepdim=True)
            unit = rotated / norm.clamp_min(torch.finfo(torch.float32).tiny)
            levels = (beta_centroids if codebook == "beta" else centroids)(part_bits, length).to(values.device)
            snapped = levels[torch.bucketize(unit.contiguous(), (levels[1:] + levels[:-1]) / 2)]
            restored.scatter_(-1, gather, (snapped * norm.to(torch.float16).float()) @ matrix.T)
    if signs == "rope_pairs":
        restored = restored * rope_pair_signs(layer_idx, features).to(values.device)
    return restored.reshape(tensor.shape).to(dtype=tensor.dtype)


def _part_matrix(layer_idx: int, length: int, signs: str, rotation: str) -> torch.Tensor:
    """Rotation of one dynamic part (r = x @ matrix): the fixed split's part rotation for this length."""
    eye = torch.eye(length, dtype=torch.float64)
    return _rotate_part(eye, layer_idx, part_signs=signs != "rope_pairs", dense=rotation == "orthogonal").float()


def _quantize_with(plan: _Plan, tensor: torch.Tensor) -> torch.Tensor:
    """Rotate (one batched matmul), quantize each segment against its own norm and codebook, rotate back.

    Same algorithm as the reference (normalize each part, rotate, nearest level, rotate back, times the
    fp16 norm), in a few GPU operations per call instead of ~100: a part's norm is taken after its
    rotation (equal up to rounding) and the nearest level comes from the midpoints (``bucketize``;
    ties go to the lower level, as argmin did).
    """
    values = tensor.float()
    with _full_fp32_matmul(values.device):
        rotated = torch.matmul(values, plan.matrix.unsqueeze(0) if plan.matrix.shape[0] > 1 else plan.matrix[0])
        pieces = []
        for start, length, levels, boundaries in plan.segments:
            part = rotated[..., start:start + length]
            norm = part.norm(dim=-1, keepdim=True)
            unit = part / norm.clamp_min(torch.finfo(torch.float32).tiny)
            index = torch.bucketize(unit.contiguous(), boundaries)
            pieces.append(levels[index] * norm.to(torch.float16).float())
        restored = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        back = plan.back.unsqueeze(0) if plan.back.shape[0] > 1 else plan.back[0]
        return torch.matmul(restored, back).to(dtype=tensor.dtype)


class _full_fp32_matmul:
    """Keep CUDA matmuls in full float32 (no TF32) inside the quantizer, whatever the global setting."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"

    def __enter__(self):
        if self.cuda:
            self.saved = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc):
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = self.saved
        return False


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


def _check_choice(name: str, value, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"turboquant_mse {name} must be one of {allowed}, got {value!r}")


def _check_bits(bits: int) -> None:
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1 or bits > 8:
        raise ValueError("turboquant_mse bits must be an integer 1..8")


METHODS["turboquant_mse"] = apply
OPTION_VALIDATORS["turboquant_mse"] = validate
WINDOW_GROUPS["turboquant_mse"] = lambda options: options.get("outlier_group", 0) or 1
