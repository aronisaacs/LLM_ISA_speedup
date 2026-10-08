"""Prefill group merging with independent sparse directional residuals.

Optimistic value-only accounting: excludes masks, norms, and group flags.
Groups are fixed adjacent blocks; the incomplete tail remains exact.
"""
from __future__ import annotations
import math
import torch
from engine.kv_compress.methods import METHODS
from engine.kv_compress.rope import apply_rope
from compression_topics.spatial.algorithms import pair_gate


def group_saving(kept: int, dim: int = 128, group_size: int = 4) -> float:
    if isinstance(group_size, bool) or group_size not in (2, 3, 4):
        raise ValueError("group_size must be 2, 3, or 4")
    if isinstance(kept, bool) or not isinstance(kept, int) or not 0 <= kept <= dim:
        raise ValueError("residual_entries must be an integer within head dimension")
    return 1 - 1 / group_size - kept / dim


def apply(tensor, *, layer_idx, target, saving, residual_entries=0,
          group_size=4, seq_start=0, rope_tables=None, **unused):
    if target not in ('k', 'v'):
        raise ValueError("target must be k or v")
    dim = tensor.shape[-1]
    maximum = group_saving(residual_entries, dim, group_size)
    if isinstance(saving, bool) or not isinstance(saving, (float, int)) or not math.isfinite(saving) or not 0 <= saving <= max(0., maximum):
        raise ValueError("saving must be finite and within the value-only representation maximum")
    if seq_start != 0 or saving == 0:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("group compression requires batch size one")
    if target == 'k' and (rope_tables is None or rope_tables.attention_scaling != 1.):
        raise ValueError("keys require pure-rotation RoPE tables")
    full = tensor.shape[-2] // group_size * group_size
    if not full:
        return tensor
    original = tensor[..., :full, :].float()
    norms = original.norm(dim=-1, keepdim=True)
    # Relative offsets align every key to the first position in its group.
    cos = sin = None
    aligned = original
    if target == 'k':
        offsets = torch.arange(full, device=tensor.device) % group_size
        cos, sin = rope_tables.cos_sin(offsets, original.dtype)
        aligned = apply_rope(original, cos, sin, inverse=True)
    directions = torch.nn.functional.normalize(aligned, dim=-1)
    grouped = directions.reshape(*directions.shape[:-2], full // group_size, group_size, dim)
    mean = grouped.mean(dim=-2, keepdim=True)
    residual = pair_gate._largest(grouped - mean, residual_entries)
    reconstruction = mean + residual
    valid = (norms.reshape(*grouped.shape[:-1], 1).squeeze(-1) > 1e-12).all(-1)
    valid &= (reconstruction.norm(dim=-1) > 1e-6).all(-1)
    reconstruction = torch.nn.functional.normalize(reconstruction, dim=-1)
    fidelity = (grouped * reconstruction).sum(-1).amin(-1).clamp(-1, 1)
    flat = fidelity.masked_fill(~valid, -float('inf')).reshape(-1)
    # Dense tails count toward the requested saving; impossible tails are logged.
    saved_bits = 16 * (group_size * dim - dim - group_size * residual_entries)
    requested = math.ceil(saving * tensor.numel() * 16 / saved_bits - 1e-10)
    count = min(requested, int(valid.sum()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count:
        mask[flat.topk(count, sorted=False).indices] = True
    mask = mask.reshape(fidelity.shape)
    restored = reconstruction.reshape_as(original) * norms
    if target == 'k':
        restored = apply_rope(restored, cos, sin, inverse=False)
    token_mask = mask.unsqueeze(-1).expand(*mask.shape, group_size).reshape(*original.shape[:-1], 1)
    out = tensor.clone()
    out[..., :full, :] = torch.where(token_mask, restored.to(tensor.dtype), tensor[..., :full, :])
    key = f'{target}_layer_{layer_idx}'
    row = pair_gate.STATS.setdefault(key, {'dense_bits': 0, 'stored_bits': 0,
        'pairs': 0, 'merged': 0, 'features': dim, 'kept': residual_entries,
        'group_size': group_size, 'accounting': 'values_only_excludes_all_metadata',
        'cosine_sum': 0., 'cosine_min': 1., 'cutoff_sum': 0., 'updates': 0,
        'shortfall_updates': 0})
    dense_bits = tensor.numel() * 16
    row['dense_bits'] += dense_bits
    row['stored_bits'] += dense_bits - count * saved_bits
    row['pairs'] += flat.numel()  # Existing measurement schema; these are groups.
    row['merged'] += count
    row['updates'] += 1
    row['shortfall_updates'] += int(count < requested)
    if count:
        chosen = fidelity[mask]
        row['cosine_sum'] += float(chosen.sum())
        row['cosine_min'] = min(row['cosine_min'], float(chosen.min()))
        row['cutoff_sum'] += float(chosen.min())
    return out


METHODS['group_calibrated'] = apply
