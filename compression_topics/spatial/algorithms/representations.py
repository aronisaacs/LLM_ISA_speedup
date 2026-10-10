"""Head-local pair/quad representations; independent of budget decision policy."""
from dataclasses import dataclass
import torch
from engine.kv_compress.rope import apply_rope
from compression_topics.spatial.algorithms.scoring import feature_scores


@dataclass
class GroupBasis:
    source: torch.Tensor
    norms: torch.Tensor
    mean: torch.Tensor
    deviation: torch.Tensor
    group_size: int
    directional: bool
    full: int
    cos: object
    sin: object


def prepare_groups(tensor, *, target, group_size, directional, rope_tables):
    if group_size not in (2, 4, 8):
        raise ValueError('representation supports adjacent pairs, quads or eights')
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError('representation requires batch-one KV tensors')
    if target not in ('k', 'v') or (target == 'k' and
            (rope_tables is None or rope_tables.attention_scaling != 1.)):
        raise ValueError('keys require pure-rotation RoPE alignment')
    full = tensor.shape[-2] // group_size * group_size
    if not full:
        raise ValueError('need at least one complete group')
    source = tensor[..., :full, :].float()
    cos = sin = None
    if target == 'k':
        offsets = torch.arange(full, device=tensor.device) % group_size
        cos, sin = rope_tables.cos_sin(offsets, torch.float32)
        source = apply_rope(source, cos, sin, inverse=True)
    norms = source.norm(dim=-1, keepdim=True)
    vectors = torch.nn.functional.normalize(source, dim=-1) if directional else source
    grouped = vectors.reshape(1, tensor.shape[1], full // group_size, group_size, tensor.shape[-1])
    mean = grouped.mean(-2, keepdim=True)
    deviation = (grouped[..., 0, :] - grouped[..., 1, :]).unsqueeze(-2) / 2 if group_size == 2 else grouped - mean
    return GroupBasis(source.reshape_as(grouped), norms.reshape(*grouped.shape[:-1], 1),
                      mean, deviation, group_size, directional, full, cos, sin)


def reconstruct(basis, residual_entries, *, dtype, selection='deviation', query_energy=None,
                indices=None, round_mean_before_residual=False, shared_mask=False):
    """Build a candidate without selecting which groups will use it.

    Legacy adaptive pairs round the mean first; fixed methods rank the original
    deviations before rounding. Preserve that numerical convention explicitly.
    """
    mean = basis.mean.to(dtype).float() if round_mean_before_residual else basis.mean
    residual = torch.zeros_like(basis.deviation)
    if residual_entries:
        if indices is None:
            scores = feature_scores(basis.deviation, selection=selection, query_energy=query_energy)
            if shared_mask and basis.group_size > 2:
                # One mask per merged group per head, chosen from all its tokens.
                scores = scores.sum(-2, keepdim=True)
            indices = scores.topk(residual_entries, dim=-1).indices
        if shared_mask and indices.shape[-2] != 1:
            raise ValueError('shared mask requires one feature-index set per group/head')
        selected = indices[..., :residual_entries]
        if shared_mask and basis.group_size > 2:
            selected = selected.expand(*basis.deviation.shape[:-1], residual_entries)
        residual.scatter_(-1, selected, basis.deviation.gather(-1, selected))
    mean, residual = mean.to(dtype).float(), residual.to(dtype).float()
    restored = torch.cat((mean + residual, mean - residual), -2) if basis.group_size == 2 else mean + residual
    valid = (basis.norms.squeeze(-1) > 1e-12).all(-1)
    if basis.directional:
        size = restored.norm(dim=-1, keepdim=True)
        valid &= (size.squeeze(-1) > 1e-6).all(-1)
        scales = (basis.norms / size.clamp_min(1e-12)).to(dtype).float()
        restored = restored * scales
        valid &= torch.isfinite(restored).all(-1).all(-1)
    return restored, valid
