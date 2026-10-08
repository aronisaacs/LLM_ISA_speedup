"""Four-token directional residual experiments; fixed configuration per KV head.

Keys use canonical pre-RoPE coordinates so shared bases have a consistent frame.
Prefill-only reconstruction simulation. fp16 means, norms, coefficients,
directions and bases are rounded before use. No packed attention implementation.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
from engine.kv_compress.methods import METHODS
from engine.kv_compress.rope import apply_rope
from compression_topics.spatial.algorithms import pair_gate

ACCOUNTING = 'pca_quad_fp16_masks_norms_bitmap_header_basis_v1'
SAMPLES = {}
_BASIS_CACHE = {}


def half(x):
    return x.to(torch.float16).float()


def grouped_directions(tensor, target, rope_tables):
    full = tensor.shape[-2] // 4 * 4
    raw = tensor[..., :full, :].float()
    norms = raw.norm(dim=-1, keepdim=True)
    tables = None
    if target == 'k':
        if rope_tables is None or rope_tables.attention_scaling != 1.:
            raise ValueError('keys require pure rotation RoPE')
        tables = rope_tables.cos_sin(torch.arange(full, device=tensor.device), torch.float32)
        raw = apply_rope(raw, *tables, inverse=True)
    unit = torch.nn.functional.normalize(raw, dim=-1)
    return unit.reshape(*unit.shape[:-2], full // 4, 4, unit.shape[-1]), norms, tables


def reconstruct(unit, kind, size, basis=None):
    """Return normalized directions and per-group worst cosine error.

    Rank-one direction is fitted locally, truncated, then coefficients refitted.
    Three independent coefficients are stored; the fourth is their negative sum.
    Shared PCA is uncentered on group residuals (their aggregate mean is zero).
    """
    dim = unit.shape[-1]
    if kind not in ('rank1', 'shared', 'separate'):
        raise ValueError('unknown residual representation')
    if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= dim:
        raise ValueError('size must be an integer in the head dimension')
    if kind != 'separate' and size == 0:
        raise ValueError('PCA requires a positive size')
    exact_mean = unit.mean(-2, keepdim=True)
    residual = unit - exact_mean
    mean = half(exact_mean)
    if kind == 'rank1':
        # Only a 4x4 eigensystem per group, rather than a 128x128 system.
        gram = residual @ residual.transpose(-1, -2)
        _, left = torch.linalg.eigh(gram)
        direction = (left[..., -1:].transpose(-1, -2) @ residual).squeeze(-2)
        direction = torch.nn.functional.normalize(direction, dim=-1)
        direction = half(pair_gate._largest(direction, size))
        denominator = direction.square().sum(-1, keepdim=True).clamp_min(1e-20)
        coeff = (residual * direction.unsqueeze(-2)).sum(-1) / denominator
        first = half(coeff[..., :3])
        coeff = torch.cat((first, -first.sum(-1, keepdim=True)), -1)
        correction = coeff.unsqueeze(-1) * direction.unsqueeze(-2)
    elif kind == 'shared':
        if basis is None or basis.shape != (dim, size):
            raise ValueError('shared basis must match dimension and rank')
        basis = half(basis.to(device=unit.device))
        # Rounded basis is not exactly orthogonal; fit its least-squares map.
        projector = torch.linalg.pinv(basis)
        coeff = half(residual @ projector.transpose(-1, -2))
        correction = coeff @ basis.transpose(-1, -2)
    else:
        correction = half(pair_gate._largest(residual, size))
    raw = mean + correction
    valid = (raw.norm(dim=-1) > 1e-6).all(-1)
    normalized = torch.nn.functional.normalize(raw, dim=-1)
    error = 1 - (unit * normalized).sum(-1).amin(-1).clamp(-1, 1)
    error = error.masked_fill(~valid, float('inf'))
    return normalized, error


def storage(kind, size, dim=128):
    """Merged-group bits, per-head header bits, and shared-basis bits.

    Five-byte header: kind uint8, size uint8, group size uint8, threshold fp16.
    Bitmap costs are separate and rounded to bytes. No implicit basis selection.
    """
    if kind not in ('rank1', 'shared', 'separate') or not 0 <= size <= dim:
        raise ValueError('invalid storage representation')
    if kind != 'separate' and size == 0:
        raise ValueError('PCA requires positive size')
    common = 16 * (dim + 4)  # mean, four original norms
    if kind == 'rank1':
        bits = common + 16 * (size + 3) + (dim if size < dim else 0)
        basis_bits = 0
    elif kind == 'shared':
        bits = common + 16 * 4 * size
        basis_bits = 16 * dim * size
    else:
        bits = common + 16 * 4 * size + (4 * dim if 0 < size < dim else 0)
        basis_bits = 0
    return bits, 40, basis_bits


def overhead(kind, size, groups, dim=128):
    _, header, basis = storage(kind, size, dim)
    return header + 8 * math.ceil(groups / 8) + basis


def load_bases(path, digest):
    key = (str(Path(path).resolve()), digest)
    if key not in _BASIS_CACHE:
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('shared basis artifact checksum mismatch')
        _BASIS_CACHE[key] = torch.load(path, map_location='cpu', weights_only=True)
    return _BASIS_CACHE[key]


def collect(tensor, *, layer_idx, target, seq_start=0, rope_tables=None,
            samples_per_chunk=32, seed=0, **unused):
    if seq_start or tensor.shape[-2] < 4:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError('PCA collection requires batch size one')
    unit, _, _ = grouped_directions(tensor, target, rope_tables)
    slot = f'{target}_layer_{layer_idx}'
    rows = SAMPLES.setdefault(slot, [])
    rng = torch.Generator().manual_seed(seed + len(rows))
    indices = torch.randperm(unit.shape[-3], generator=rng)[:samples_per_chunk].to(tensor.device)
    rows.append(unit[0].index_select(1, indices).half().cpu())
    return tensor


def apply(tensor, *, layer_idx, target, head_settings, seq_start=0,
          rope_tables=None, basis_path=None, basis_digest=None,
          accounting=ACCOUNTING, **unused):
    if accounting != ACCOUNTING or target not in ('k', 'v'):
        raise ValueError('invalid PCA accounting or target')
    if seq_start or tensor.shape[-2] < 4:
        return tensor
    if tensor.shape[0] != 1 or len(head_settings) != tensor.shape[1]:
        raise ValueError('PCA requires batch one and one fixed setting per KV head')
    unit, norms, tables = grouped_directions(tensor, target, rope_tables)
    full = unit.shape[-3] * 4
    output = tensor.clone()
    bases = None
    for head, settings in enumerate(head_settings):
        if settings is None:  # Entire head stays dense, needs no format metadata.
            continue
        kind, size, threshold = settings['kind'], settings['size'], settings['threshold']
        if not math.isfinite(threshold) or not 0 <= threshold <= 2:
            raise ValueError('error threshold must be finite and in [0, 2]')
        basis = None
        if kind == 'shared':
            if bases is None:
                bases = load_bases(basis_path, basis_digest)
            basis = bases[f'{target}_layer_{layer_idx}'][head, :, :size]
        reconstructed, error = reconstruct(unit[0, head], kind, size, basis)
        valid_norm = norms[0, head].reshape(-1, 4).amin(-1) > 1e-12
        # No runtime top-k, target-budget adjustment, or per-group size selection.
        threshold = float(torch.tensor(threshold).half())
        merge = (error <= threshold) & valid_norm
        restored = reconstructed.reshape(full, -1) * half(norms[0, head])
        if target == 'k':
            restored = apply_rope(restored[None, None], *tables, inverse=False)[0, 0]
        token_mask = merge.repeat_interleave(4).unsqueeze(-1)
        output[0, head, :full] = torch.where(token_mask, restored.to(tensor.dtype), tensor[0, head, :full])
        groups, dim = merge.numel(), tensor.shape[-1]
        count = int(merge.sum())
        merged, _, basis_bits = storage(kind, size, dim)
        dense_bits = tensor[0, head].numel() * 16
        fixed = overhead(kind, size, groups, dim)
        slot = f'{target}_layer_{layer_idx}_head_{head}'
        row = pair_gate.STATS.setdefault(slot, {'dense_bits': 0, 'stored_bits': 0,
            'pairs': 0, 'merged': 0, 'features': dim, 'kept': size, 'group_size': 4,
            'accounting': ACCOUNTING, 'cosine_sum': 0., 'cosine_min': 1.,
            'cutoff_sum': 0., 'updates': 0, 'shortfall_updates': 0,
            'basis_bits': 0, 'fixed_metadata_bits': 0})
        row['dense_bits'] += dense_bits
        row['stored_bits'] += dense_bits - count * (4 * dim * 16 - merged) + fixed
        row['pairs'] += groups
        row['merged'] += count
        row['basis_bits'] += basis_bits  # Once per head per fresh prompt cache.
        row['fixed_metadata_bits'] += fixed - basis_bits
        row['updates'] += 1
        if count:
            cosines = 1 - error[merge]
            row['cosine_sum'] += float(cosines.sum())
            row['cosine_min'] = min(row['cosine_min'], float(cosines.min()))
            row['cutoff_sum'] += float(cosines.min())
    return output


METHODS['pca_quad_collect'] = collect
METHODS['pca_quad'] = apply
