"""Pair-local reconstruction/storage prototype with a feedback price controller.

Registered for prefill reconstruction in the model pipeline; decode appends stay dense. Each choice applies across KV heads,
but means/residuals remain within heads. Only the initial 32-token window uses
a shared price search. Afterward decisions inspect the current pair and state.
"""
from dataclasses import dataclass
import math
import torch
from compression_topics.spatial.algorithms.storage import vector_group_bits, metadata_bits
from engine.kv_compress.rope import apply_rope

RESIDUALS = (0, 8, 16, 32)
ACCOUNTING = 'online_pairs_bf16_scales_bitmap_shared_3bit_tag_v1'

@dataclass(frozen=True)
class Control:
    startup_tokens: int = 32
    memory_pairs: float = 10.
    recent_deadband: float = .02
    cumulative_deadband: float = .03
    recent_gain: float = 4.
    cumulative_gain: float = 8.
    max_log_step: float = .2
    price_span: float = 1e4

    def __post_init__(self):
        if self.startup_tokens < 2 or self.startup_tokens % 2:
            raise ValueError('startup must contain complete pairs')
        values = (self.memory_pairs, self.max_log_step, self.price_span)
        if any(not math.isfinite(x) or x <= 0 for x in values) or self.price_span <= 1:
            raise ValueError('invalid controller memory/bounds')
        if any(not math.isfinite(x) or x < 0 for x in
               (self.recent_deadband, self.cumulative_deadband, self.recent_gain, self.cumulative_gain)):
            raise ValueError('invalid feedback parameters')

@dataclass
class PairTable:
    errors: list  # [pair, option], mean relative squared reconstruction error across heads/tokens
    payload_bits: list  # dense, merged 0/8/16/32, excludes shared tags and header
    dense_pair_bits: int
    tail_bits: int


def pair_table(tensor, *, target, rope_tables=None, return_reconstruction=False):
    """Vectorized option preparation; every row depends on its own pair only."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < max(RESIDUALS):
        raise ValueError('need batch-one [1, KV heads, tokens, head_dim >= 32]')
    if target not in ('k', 'v') or (target == 'k' and
            (rope_tables is None or rope_tables.attention_scaling != 1.)):
        raise ValueError('keys require pure-rotation RoPE; target must be k/v')
    heads, length, dim = tensor.shape[1:]
    full = length // 2 * 2
    if not full:
        raise ValueError('need at least one complete pair')
    source = tensor[..., :full, :].float()
    if target == 'k':
        cos, sin = rope_tables.cos_sin(torch.arange(full, device=tensor.device) % 2, torch.float32)
        source = apply_rope(source, cos, sin, inverse=True)
    grouped = source.reshape(1, heads, full // 2, 2, dim)
    norms = grouped.norm(dim=-1, keepdim=True)
    unit = torch.nn.functional.normalize(grouped, dim=-1)
    mean = unit.mean(-2, keepdim=True).to(torch.bfloat16).float()
    delta = (unit[..., 0, :] - unit[..., 1, :]) / 2
    indices = delta.square().topk(max(RESIDUALS), dim=-1).indices
    errors = [torch.zeros(full // 2, device=tensor.device)]
    reconstructions = [grouped] if return_reconstruction else None
    for residual_count in RESIDUALS:
        residual = torch.zeros_like(delta)
        if residual_count:
            selected = indices[..., :residual_count]
            residual.scatter_(-1, selected, delta.gather(-1, selected))
        residual = residual.to(torch.bfloat16).float()
        direction = torch.cat((mean + residual.unsqueeze(-2), mean - residual.unsqueeze(-2)), -2)
        size = direction.norm(dim=-1, keepdim=True)
        scales = (norms / size.clamp_min(1e-12)).to(torch.bfloat16).float()
        restored = direction * scales
        error = ((restored-grouped).square().sum(-1) / grouped.square().sum(-1).clamp_min(1e-12)).mean((0,1,3))
        valid = ((size.squeeze(-1) > 1e-6) & (norms.squeeze(-1) > 1e-12)).all(-1).all((0,1))
        error = error.masked_fill(~valid, float('inf'))
        errors.append(error)
        if return_reconstruction:reconstructions.append(restored)
    dense = heads * 2 * dim * 16
    table = PairTable(torch.stack(errors,-1).cpu().tolist(),
                     [dense] + [heads * vector_group_bits(r, dim, 2, include_norms=True) for r in RESIDUALS],
                     dense, (length-full)*heads*dim*16)
    return (table,reconstructions) if return_reconstruction else table


def select(error, rates, price, allowed):
    return min(allowed, key=lambda i: (error[i] + price*rates[i], i))


def initial_price(table, rows, target_saving, allowed):
    """Fit startup storage, preferring the undercompressed side of a breakpoint.

    Exact targets are kept. If choices jump past the target, take the last
    lower-compression price rather than oversaving. Only startup rows are seen.
    """
    dense = len(rows)*table.dense_pair_bits
    ceiling_bits = sum(min(table.payload_bits[i] for i in allowed if math.isfinite(e[i])) for e in rows)
    overhead = metadata_bits(len(rows), 3)
    if ceiling_bits + overhead > dense*(1-target_saving)+1e-8:
        raise ValueError('startup cannot meet target with available pair formats')
    rates = [b/table.dense_pair_bits for b in table.payload_bits]
    def stored(price):
        return sum(table.payload_bits[select(e,rates,price,allowed)] for e in rows)+overhead
    high=1e-6
    while stored(high)>dense*(1-target_saving):
        high*=4
        if high>1e12:
            raise ValueError('cannot initialize price')
    low=0.
    for _ in range(60):
        middle=(low+high)/2
        if stored(middle)<=dense*(1-target_saving):high=middle
        else:low=middle
    # Budget jumps may span several tied pairs. Favor information preservation,
    # even if the undercompressed setting is farther away from the target.
    exact = abs(stored(high) - dense*(1-target_saving)) <= 1e-8
    return high if exact else low


def outside(error, band):
    return math.copysign(max(abs(error)-band, 0.), error)


def asymmetric_outside(error, undercompression_band, overcompression_band):
    """Positive stored-fraction error means undercompression, negative means overcompression."""
    return outside(error, undercompression_band if error >= 0 else overcompression_band)


def effective_bands(table, allowed, pairs, control):
    """Undercompression tolerance plus one pair's format-dependent storage impact.

    Tags have identical width for all options, so their packing does not change
    the difference between two format decisions. The allowance shrinks as 1/N.
    """
    span = (max(table.payload_bits[i] for i in allowed) -
            min(table.payload_bits[i] for i in allowed)) / table.dense_pair_bits
    slack = span / pairs
    return control.recent_deadband + slack, control.cumulative_deadband + slack, slack


def run(table, target_saving, *, feedback=True, fixed_residual=None, control=Control()):
    if not math.isfinite(target_saving) or not 0<target_saving<.5:
        raise ValueError('pair target must be positive and below 50%')
    allowed = list(range(5)) if fixed_residual is None else [0, RESIDUALS.index(fixed_residual)+1]
    warmup=min(control.startup_tokens//2,len(table.errors))
    price=initial_price(table,table.errors[:warmup],target_saving,allowed)
    initial=price
    anchor=max(initial,1e-12)
    rates=[b/table.dense_pair_bits for b in table.payload_bits]
    target=1-target_saving
    alpha=1-math.exp(-1/control.memory_pairs)
    payload=0;previous_stored=0;ema=target;history=[];total_error=0.;hits=0
    for n,error in enumerate(table.errors,1):
        # No access to any later row: the only exception is the explicit startup initialization above.
        used_price=price
        choice=select(error,rates,used_price,allowed)
        payload+=table.payload_bits[choice]
        stored=payload+metadata_bits(n,3)
        total_error+=error[choice]
        fraction=stored/(n*table.dense_pair_bits)
        increment=(stored-previous_stored)/table.dense_pair_bits
        recent_band,cumulative_band,pair_slack=effective_bands(table,allowed,n,control)
        if n==warmup:
            ema=fraction  # Initialize from the actual committed 32-token window.
        elif n>warmup:
            ema=(1-alpha)*ema+alpha*increment
            if feedback:
                correction=control.recent_gain*asymmetric_outside(ema-target,recent_band,control.recent_deadband)
                correction+=control.cumulative_gain*asymmetric_outside(fraction-target,cumulative_band,control.cumulative_deadband)
                change=max(-control.max_log_step,min(control.max_log_step,correction))
                # A zero-price dense startup must still be able to begin correcting
                # undercompression. Seed the first positive adjustment at a finite floor.
                new=(price if price>0 else anchor/control.price_span)*math.exp(change)
                price=max(0.,min(anchor*control.price_span,new))
                if price>0:price=max(anchor/control.price_span,price)
                hits+=int(price!=new)
        history.append({'tokens':2*n,'format':'dense' if choice==0 else f'r{RESIDUALS[choice-1]}',
                        'price_used':used_price,'price_next':price,'stored_bits':stored,
                        'cumulative_saving':1-fraction,'recent_saving':1-ema,
                        'recent_undercompression_band':recent_band,'recent_overcompression_band':control.recent_deadband,
                        'cumulative_undercompression_band':cumulative_band,'cumulative_overcompression_band':control.cumulative_deadband,
                        'allowed_saving_min':target_saving-cumulative_band,'allowed_saving_max':target_saving+control.cumulative_deadband,
                        'one_pair_storage_slack':pair_slack,
                        'pair_relative_error':error[choice], 'running_relative_error':total_error/n})
        previous_stored=stored
    final_stored=stored+table.tail_bits
    final_dense=len(table.errors)*table.dense_pair_bits+table.tail_bits
    settled=history[warmup:]
    return {'target_saving':target_saving,'measured_saving':1-final_stored/final_dense,
            'mean_relative_squared_error':total_error/len(table.errors),'initial_price':initial,
            'price_bound_hits':hits,'startup_saving':history[warmup-1]['cumulative_saving'],
            'fraction_post_startup_within_three_points':sum(abs(r['cumulative_saving']-target_saving)<=.03 for r in settled)/len(settled) if settled else None,
            'early_checkpoints':[r for r in history if r['tokens'] in (32,48,64,96,128,192,256)],
            'early_max_cumulative_deviation':max((abs(r['cumulative_saving']-target_saving) for r in history[warmup-1:128]),default=None),
            'fraction_post_startup_within_effective_band':sum(r['allowed_saving_min']<=r['cumulative_saving']<=r['allowed_saving_max'] for r in settled)/len(settled) if settled else None,
            'formats':{name:sum(r['format']==name for r in history) for name in ['dense','r0','r8','r16','r32']},
            'stored_bits':final_stored,'dense_bits':final_dense,'history':history}


def apply(tensor, *, layer_idx, target, saving, feedback=True, fixed_residual=None,
          control=None, seq_start=0, rope_tables=None, accounting=ACCOUNTING):
    """Compress prefill using local controller choices; no new decode compression."""
    from engine.kv_compress.metrics import STATS
    if accounting != ACCOUNTING:
        raise ValueError('unsupported online-pair accounting')
    if not isinstance(feedback,bool):
        raise ValueError('feedback must be boolean')
    settings=Control(**(control or {}))
    if seq_start:
        return tensor
    table,reconstructions=pair_table(tensor,target=target,rope_tables=rope_tables,return_reconstruction=True)
    result=run(table,saving,feedback=feedback,fixed_residual=fixed_residual,control=settings)
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


from engine.kv_compress.methods import METHODS
METHODS['online_pairs']=apply
