"""Fresh presentation study: RoPE-aligned fixed pairs/quads, no dynamic formats.

Raw pairs are the baseline. Directional groups store a 16-bit norm-restoration scale per token,
one signed residual/mask per pair or one residual/mask per quad token. Groups
rank by worst relative squared reconstruction error. Keys always align RoPE;
values never rotate. This is a prefill reconstruction/storage simulation.
"""
import math
import torch

from compression_topics.spatial.algorithms.storage import vector_group_bits, metadata_bits
from compression_topics.spatial.algorithms.representations import prepare_groups, reconstruct
from compression_topics.spatial.algorithms.decisions import rank_groups
from compression_topics.spatial.algorithms.scoring import query_energy, reconstruction_error
from engine.kv_compress.methods import METHODS
from engine.kv_compress.metrics import STATS
from engine.kv_compress.rope import apply_rope

ACCOUNTING = 'presentation_spatial_fp16_bitmap_norms_header_v1'


def merged_bits(kept, dim=128, group_size=2, directional=True):
    return vector_group_bits(kept, dim, group_size, include_norms=directional)


def maximum_saving(kept, dim=128, group_size=2, directional=True):
    return 1 - (merged_bits(kept, dim, group_size, directional) + 1) / (16 * group_size * dim)


def apply_components(tensor, *, layer_idx, target, saving, residual_entries=0, group_size=2,
          directional=True, select_by='deviation', query_weights=None,
          query_weights_sha256=None, seq_start=0, rope_tables=None, accounting=ACCOUNTING,
                     distortion='relative', token_weights=None):
    if accounting != ACCOUNTING:
        raise ValueError('unsupported clean-study accounting')
    if target not in ('k', 'v') or group_size not in (2, 4) or not isinstance(directional, bool):
        raise ValueError('need K/V, group size 2/4 and boolean directional')
    dim = tensor.shape[-1]
    if isinstance(residual_entries, bool) or not isinstance(residual_entries, int) or not 0 <= residual_entries <= dim:
        raise ValueError('residual count must fit the head dimension')
    if not directional and (group_size != 2 or residual_entries or select_by != 'deviation'):
        raise ValueError('raw baseline supports pairs without residuals only')
    if select_by not in ('deviation', 'query'):
        raise ValueError('select_by must be deviation or query')
    if (select_by == 'query' or distortion == 'query') and (target != 'k' or query_weights is None or query_weights_sha256 is None):
        raise ValueError('query selection needs key weights and their checksum')
    if select_by != 'query' and distortion != 'query' and (query_weights is not None or query_weights_sha256 is not None):
        raise ValueError('plain selection must not carry query weights')
    ceiling = maximum_saving(residual_entries, dim, group_size, directional)
    if isinstance(saving, bool) or not isinstance(saving, (int, float)) or not math.isfinite(saving) or not 0 <= saving <= ceiling:
        raise ValueError('saving exceeds this representation ceiling')
    if seq_start or saving == 0:
        return tensor  # Decode appends remain dense.
    if tensor.shape[0] != 1:
        raise ValueError('clean spatial study requires batch size one')
    if target == 'k' and (rope_tables is None or rope_tables.attention_scaling != 1.):
        raise ValueError('all clean key variants require pure-rotation RoPE alignment')
    basis = prepare_groups(tensor, target=target, group_size=group_size,
                           directional=directional, rope_tables=rope_tables)
    full = basis.full
    original = tensor[..., :full, :].float()
    cos, sin = basis.cos, basis.sin
    weights = query_energy(query_weights, layer_idx, query_weights_sha256, tensor) if select_by == 'query' or distortion == 'query' else None
    reconstructed, valid = reconstruct(basis, residual_entries, dtype=tensor.dtype,
                                       selection=select_by, query_energy=weights)
    source = basis.source
    grouped_weights = None if token_weights is None else token_weights[..., :full].reshape(*source.shape[:-1])
    error = reconstruction_error(source, reconstructed, distortion=distortion, query_energy=weights,
                                 token_weights=grouped_weights).amax(-1)
    mask, count, requested, overhead, saving_per_group = rank_groups(
        error, valid, saving=saving, dense_bits=tensor.numel()*16,
        dense_group_bits=16*group_size*dim,
        merged_group_bits=merged_bits(residual_entries, dim, group_size, directional))
    restored = reconstructed.reshape_as(original)
    if target == 'k':
        restored = apply_rope(restored, cos, sin, inverse=False)
    token_mask = mask.unsqueeze(-1).expand(*mask.shape, group_size).reshape(1, tensor.shape[1], full, 1)
    out = tensor.clone()
    out[..., :full, :] = torch.where(token_mask, restored.to(tensor.dtype), tensor[..., :full, :])
    cosine = torch.nn.functional.cosine_similarity(source, reconstructed, dim=-1).amin(-1)
    row = STATS.setdefault(f'{target}_layer_{layer_idx}', {
        'dense_bits': 0, 'stored_bits': 0, 'pairs': 0, 'merged': 0, 'features': dim,
        'kept': residual_entries, 'group_size': group_size, 'accounting': ACCOUNTING,
        'metadata_bits': 0, 'norm_bits': 0, 'residual_mask_bits': 0,
        'cosine_sum': 0., 'cosine_min': 1., 'cutoff_sum': 0., 'updates': 0, 'shortfall_updates': 0})
    dense_bits = tensor.numel() * 16
    row['dense_bits'] += dense_bits
    row['stored_bits'] += dense_bits - count * saving_per_group + overhead
    row['metadata_bits'] += overhead
    row['norm_bits'] += count * group_size * 16 if directional else 0
    row['residual_mask_bits'] += count * (1 if group_size == 2 else group_size) * dim if residual_entries else 0
    row['pairs'] += error.numel()
    row['merged'] += count
    row['updates'] += 1
    row['shortfall_updates'] += int(count < requested)
    if count:
        row['cosine_sum'] += float(cosine[mask].sum())
        row['cosine_min'] = min(row['cosine_min'], float(cosine[mask].min()))
        row['cutoff_sum'] += float(cosine[mask].min())
    return out


def apply(tensor, *, layer_idx, target, saving, residual_entries=0, group_size=2,
          directional=True, select_by='deviation', query_weights=None,
          query_weights_sha256=None, seq_start=0, rope_tables=None, accounting=ACCOUNTING):
    """Compatibility entry point preserving existing study signatures."""
    return apply_components(tensor, layer_idx=layer_idx, target=target, saving=saving,
                            residual_entries=residual_entries, group_size=group_size,
                            directional=directional, select_by=select_by, query_weights=query_weights,
                            query_weights_sha256=query_weights_sha256, seq_start=seq_start,
                            rope_tables=rope_tables, accounting=accounting)


METHODS['presentation_spatial'] = apply
