"""Pair-local reconstruction/storage prototype with a feedback price controller.

Registered for prefill reconstruction in the model pipeline; decode appends stay dense. Each choice applies across KV heads,
but means/residuals remain within heads. In online mode only the initial 32-token window uses
a shared price search. Afterward decisions inspect the current pair and state.
Offline mode fits its price over all pairs before committing formats.
"""
import torch
from compression_topics.spatial.algorithms.storage import vector_group_bits, metadata_bits
from engine.kv_compress.rope import apply_rope

RESIDUALS = (0, 8, 16, 32)
ACCOUNTING = 'online_pairs_bf16_scales_bitmap_shared_3bit_tag_v1'

# Compatibility exports keep existing studies, configs and results unchanged.
from compression_topics.spatial.algorithms.decisions import (
    Control, PairTable, select, initial_price, outside, asymmetric_outside,
    effective_bands, run, run_offline,
)
from compression_topics.spatial.algorithms.representations import prepare_groups, reconstruct
from compression_topics.spatial.algorithms.scoring import feature_scores, reconstruction_error


def pair_table(tensor, *, target, rope_tables=None, return_reconstruction=False):
    return build_pair_table(tensor, target=target, rope_tables=rope_tables,
                            return_reconstruction=return_reconstruction)


def build_pair_table(tensor, *, target, rope_tables=None, return_reconstruction=False,
                     selection='deviation', distortion='relative', query_energy=None, token_weights=None):
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < max(RESIDUALS):
        raise ValueError('need batch-one [1, KV heads, tokens, head_dim >= 32]')
    basis = prepare_groups(tensor, target=target, group_size=2, directional=True, rope_tables=rope_tables)
    heads, length, dim = tensor.shape[1:]
    full = basis.full
    # Preserve the legacy adaptive top-k ordering by preparing the full menu once.
    indices = feature_scores(basis.deviation, selection=selection, query_energy=query_energy).topk(max(RESIDUALS), dim=-1).indices
    errors = [torch.zeros(full // 2, device=tensor.device)]
    reconstructions = [basis.source] if return_reconstruction else None
    grouped_weights = None if token_weights is None else token_weights[..., :full].reshape(1, heads, full // 2, 2)
    for residual_count in RESIDUALS:
        restored, valid = reconstruct(basis, residual_count, dtype=torch.bfloat16,
                                      indices=indices, round_mean_before_residual=True)
        error = reconstruction_error(basis.source, restored, distortion=distortion,
                                     query_energy=query_energy, token_weights=grouped_weights).mean((0,1,3))
        error = error.masked_fill(~valid.all((0,1)), float('inf'))
        errors.append(error)
        if return_reconstruction:
            reconstructions.append(restored)
    dense = heads * 2 * dim * 16
    table = PairTable(torch.stack(errors,-1).cpu().tolist(),
                     [dense] + [heads * vector_group_bits(r, dim, 2, include_norms=True) for r in RESIDUALS],
                     dense, (length-full)*heads*dim*16)
    return (table,reconstructions) if return_reconstruction else table


def apply_components(tensor, *, layer_idx, target, saving, feedback=True, fixed_residual=None,
          control=None, seq_start=0, rope_tables=None, accounting=ACCOUNTING, decision="online",
                     selection="deviation", distortion="relative", query_energy=None, token_weights=None):
    """Compress prefill using online controller or offline reference; appends dense."""
    from engine.kv_compress.metrics import STATS
    if accounting != ACCOUNTING:
        raise ValueError('unsupported online-pair accounting')
    if not isinstance(feedback,bool):
        raise ValueError('feedback must be boolean')
    if decision not in ('online', 'offline'):
        raise ValueError('decision must be online or offline')
    if decision == 'offline' and (fixed_residual is not None or control is not None):
        raise ValueError('offline reference has no fixed residual/controller settings')
    settings=Control(**(control or {}))
    if seq_start:
        return tensor
    table,reconstructions=build_pair_table(tensor,target=target,rope_tables=rope_tables,return_reconstruction=True,
                                           selection=selection,distortion=distortion,query_energy=query_energy,token_weights=token_weights)
    result=run_offline(table,saving) if decision == 'offline' else run(table,saving,feedback=feedback,fixed_residual=fixed_residual,control=settings)
    names=['dense']+[f'r{r}' for r in RESIDUALS]
    choice=torch.tensor([names.index(r['format']) for r in result['history']],device=tensor.device)
    selected=torch.zeros_like(reconstructions[0])
    for i,option in enumerate(reconstructions):
        selected=torch.where((choice==i).reshape(1,1,-1,1,1),option,selected)
    full=2*len(table.errors)
    restored=selected.reshape(1,tensor.shape[1],full,tensor.shape[-1])
    if target=='k':
        cos,sin=rope_tables.cos_sin(torch.arange(full,device=tensor.device)%2,torch.float32)
        restored=apply_rope(restored,cos,sin,inverse=False)
    merged=(choice>0).repeat_interleave(2).reshape(1,1,full,1)
    out=tensor.clone()
    out[...,:full,:]=torch.where(merged,restored.to(tensor.dtype),tensor[...,:full,:])
    heads,dim=tensor.shape[1],tensor.shape[-1]
    count=sum(result['formats'][f'r{r}'] for r in RESIDUALS)
    cosine=torch.nn.functional.cosine_similarity(tensor[...,:full,:].float(),out[...,:full,:].float(),dim=-1)
    pair_cosine=cosine.reshape(1,heads,-1,2).amin((0,1,3))
    valid=pair_cosine[choice>0]
    row=STATS.setdefault(f'{target}_layer_{layer_idx}',{'accounting':ACCOUNTING,'dense_bits':0,'stored_bits':0,
        'pairs':0,'merged':0,'features':dim,'metadata_bits':0,'norm_bits':0,'residual_mask_bits':0,
        'cosine_sum':0.,'cosine_min':1.,'cutoff_sum':0.,'updates':0,'shortfall_updates':0,
        'controller_bound_hits':0,'max_final_target_deviation':0.})
    row['dense_bits']+=result['dense_bits'];row['stored_bits']+=result['stored_bits']
    row['pairs']+=len(table.errors);row['merged']+=count
    row['metadata_bits']+=metadata_bits(len(table.errors),3)
    row['norm_bits']+=count*heads*2*16
    row['residual_mask_bits']+=sum(result['formats'][f'r{r}']*heads*dim for r in RESIDUALS if r)
    row['updates']+=1;row['controller_bound_hits']+=result['price_bound_hits']
    row['max_final_target_deviation']=max(row['max_final_target_deviation'],abs(result['measured_saving']-saving))
    if len(valid):
        row['cosine_sum']+=float(valid.sum());row['cosine_min']=min(row['cosine_min'],float(valid.min()))
        row['cutoff_sum']+=float(valid.min())
    return out


def apply(tensor, *, layer_idx, target, saving, feedback=True, fixed_residual=None,
          control=None, seq_start=0, rope_tables=None, accounting=ACCOUNTING, decision="online"):
    """Compatibility entry point for existing pair studies and saved identities."""
    return apply_components(tensor, layer_idx=layer_idx, target=target, saving=saving,
                            feedback=feedback, fixed_residual=fixed_residual, control=control,
                            seq_start=seq_start, rope_tables=rope_tables, accounting=accounting, decision=decision)


from engine.kv_compress.methods import METHODS
METHODS['online_pairs']=apply
