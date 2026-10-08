"""Prefill-only directional pair compression at a target byte saving.

Keys align the second token one RoPE step backward; values need no rotation.
Both save original norms separately (their bytes are intentionally excluded).
One shared directional mean plus one sparse signed half-difference per pair.
Pairs rank by the worse of their two actual reconstruction cosines.
"""
from __future__ import annotations

import math
import torch
from engine.kv_compress.methods import METHODS
from compression_topics.spatial.algorithms.residual import largest_residual
from engine.kv_compress.metrics import STATS
from compression_topics.spatial.algorithms.storage import vector_group_bits
from compression_topics.spatial.algorithms.pair_rank import _shift_rope


def pair_saving(kept: int, dim: int = 128) -> float:
    if isinstance(kept, bool) or not isinstance(kept, int) or not 0 <= kept <= dim:
        raise ValueError("residual_entries must be an integer within head dimension")
    return 1 - vector_group_bits(kept, dim, 2, include_norms=False) / (32 * dim)


def apply(tensor, *, layer_idx, target, saving, residual_entries=0, seq_start=0, rope_tables=None, **unused):
    if target not in {"k", "v"}:
        raise ValueError("target must be k or v")
    if not isinstance(saving, (int, float)) or isinstance(saving, bool) or not math.isfinite(saving):
        raise ValueError("saving must be finite")
    dim = tensor.shape[-1]
    maximum = pair_saving(residual_entries, dim)
    if not 0 <= saving <= maximum + 1e-12:
        raise ValueError(f"saving {saving} exceeds residual representation maximum {maximum}")
    if seq_start != 0:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("calibrated pair compression requires batch size 1")
    if saving == 0:
        return tensor
    if target == "k" and (rope_tables is None or rope_tables.attention_scaling != 1.):
        raise ValueError("keys require pure-rotation RoPE tables")
    full = tensor.shape[-2] // 2 * 2
    if full == 0:
        return tensor
    first = tensor[..., :full:2, :].float()
    second = tensor[..., 1:full:2, :].float()
    # Save the norms as stored, before numerical rotation error.
    na, nb = first.norm(dim=-1, keepdim=True), second.norm(dim=-1, keepdim=True)
    if target == "k":
        second = _shift_rope(second, rope_tables, inverse=True)
    a = torch.nn.functional.normalize(first, dim=-1)
    b = torch.nn.functional.normalize(second, dim=-1)
    mean, delta = (a + b) / 2, (a - b) / 2
    residual = largest_residual(delta, residual_entries)
    ra, rb = mean + residual, mean - residual
    valid = (na.squeeze(-1) > 1e-12) & (nb.squeeze(-1) > 1e-12)
    valid &= (ra.norm(dim=-1) > 1e-6) & (rb.norm(dim=-1) > 1e-6)
    ra, rb = torch.nn.functional.normalize(ra, dim=-1), torch.nn.functional.normalize(rb, dim=-1)
    fidelity = torch.minimum((a * ra).sum(-1), (b * rb).sum(-1)).clamp(-1, 1)
    flat = fidelity.masked_fill(~valid, -float("inf")).reshape(-1)
    # Budget is relative to the selected tensor bytes; odd tails remain dense.
    requested = math.ceil(saving * tensor.numel() / (2 * dim * maximum) - 1e-10)
    count = min(requested, int(valid.sum()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count:
        indices = flat.topk(count, largest=True, sorted=False).indices
        mask[indices] = True
    mask = mask.reshape(fidelity.shape).unsqueeze(-1)
    ra, rb = ra * na, rb * nb
    if target == "k":
        rb = _shift_rope(rb, rope_tables, inverse=False)
    out = tensor.clone()
    out[..., :full:2, :] = torch.where(mask, ra.to(tensor.dtype), tensor[..., :full:2, :])
    out[..., 1:full:2, :] = torch.where(mask, rb.to(tensor.dtype), tensor[..., 1:full:2, :])
    # Slot-specific stats support mixed residual sizes and measured budget reporting.
    key = f"{target}_layer_{layer_idx}"
    row = STATS.setdefault(key, {"dense_bits": 0, "stored_bits": 0, "pairs": 0,
        "merged": 0, "features": dim, "kept": residual_entries, "cosine_sum": 0.,
        "cosine_min": 1., "cutoff_sum": 0., "updates": 0, "shortfall_updates": 0})
    dense_bits = tensor.numel() * 16
    row["dense_bits"] += dense_bits
    row["stored_bits"] += dense_bits - count * round(32 * dim * maximum)
    row["pairs"] += flat.numel()
    row["merged"] += count
    row["updates"] += 1
    row["shortfall_updates"] += int(count < requested)
    if count:
        chosen = fidelity[mask.squeeze(-1)]
        row["cosine_sum"] += float(chosen.sum())
        row["cosine_min"] = min(row["cosine_min"], float(chosen.min()))
        row["cutoff_sum"] += float(chosen.min())
    return out


METHODS["pair_calibrated"] = apply
