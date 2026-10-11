"""Representation-independent budget decisions for adaptive pair tables.

The online controller sees the startup window, then the current row and state.
Offline fits the whole available table. Neither policy builds representations
or knows how residual features are selected.
"""
from dataclasses import dataclass
import math
import torch
from compression_topics.spatial.algorithms.storage import metadata_bits

RESIDUALS = (0, 8, 16, 32)

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


def rank_groups(error, valid, *, saving, dense_bits, dense_group_bits, merged_group_bits):
    """Fixed-format whole-prefill ranking, independent of reconstruction math."""
    flat = error.masked_fill(~valid, float('inf')).reshape(-1)
    saving_per_group = dense_group_bits - merged_group_bits
    overhead = metadata_bits(flat.numel())
    requested = math.ceil((saving * dense_bits + overhead) / saving_per_group - 1e-10)
    count = min(requested, int(valid.sum()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count:
        mask[flat.topk(count, largest=False, sorted=False).indices] = True
    return mask.reshape(error.shape), count, requested, overhead, saving_per_group


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


def run_offline(table, target_saving):
    """Whole-prefill price search with discrete boundary repair.

    Same five representations/error objective as online. This is a practical
    offline reference, not an exact multiple-choice knapsack/accuracy optimum.
    It sees every pair before committing any choices.
    """
    if not math.isfinite(target_saving) or not 0 < target_saving < .5:
        raise ValueError('pair target must be positive and below 50%')
    n = len(table.errors)
    if not n:
        raise ValueError('need complete pairs')
    dense = n * table.dense_pair_bits + table.tail_bits
    overhead = metadata_bits(n, 3) + table.tail_bits
    limit = dense * (1 - target_saving) - overhead
    errors = torch.tensor(table.errors, dtype=torch.float64)
    bits = torch.tensor(table.payload_bits, dtype=torch.float64)
    rates = bits / table.dense_pair_bits
    minimum = bits.expand_as(errors).masked_fill(~torch.isfinite(errors), float('inf')).amin(-1).sum().item()
    if minimum > limit + 1e-8:
        raise ValueError('offline target cannot be met with available pair formats')
    def choices(price):
        result = (errors + price * rates).argmin(-1)
        return result, bits[result].sum().item()
    high = 1e-6
    while choices(high)[1] > limit:
        high *= 4
        if high > 1e12:
            raise ValueError('cannot fit offline price')
    low = 0.
    for _ in range(60):
        middle = (low + high) / 2
        if choices(middle)[1] <= limit:
            high = middle
        else:
            low = middle
    lower, payload = choices(low)
    upper, _ = choices(high)
    selected = lower.tolist()
    # Split simultaneous price-breakpoint transitions rather than switching
    # every tied pair at once. Stop after reaching the required byte saving.
    changes = []
    for pair, (before, after) in enumerate(zip(selected, upper.tolist())):
        gain = table.payload_bits[before] - table.payload_bits[after]
        if gain > 0:
            cost = table.errors[pair][after] - table.errors[pair][before]
            changes.append((cost / gain, pair, after, gain))
    for _, pair, after, gain in sorted(changes):
        if payload <= limit + 1e-8:
            break
        selected[pair] = after
        payload -= gain
    if payload > limit + 1e-8:
        raise ValueError('offline boundary repair missed target')
    names = ['dense'] + [f'r{r}' for r in RESIDUALS]
    return {'target_saving': target_saving, 'measured_saving': 1 - (payload + overhead) / dense,
            'stored_bits': int(payload + overhead), 'dense_bits': dense,
            'mean_relative_squared_error': sum(table.errors[j][i] for j, i in enumerate(selected)) / n,
            'price_bound_hits': 0, 'initial_price': high,
            'formats': {name: selected.count(i) for i, name in enumerate(names)},
            'history': [{'tokens': 2*(j+1), 'format': names[i]} for j, i in enumerate(selected)]}


def run_head_pairs(table, target_saving, *, feedback=True, fixed_residual=None, control=Control(), offline=False):
    """Choose each head independently; update one controller per token pair.

    Rows are pair-major [pair][head][format]. The shared price is initialized
    from the first startup_tokens, never startup_tokens per head. Tags are per
    head/pair with one packed slot header. Memory remains measured in tokens.
    """
    if not math.isfinite(target_saving) or not 0<target_saving<.5:
        raise ValueError('pair target must be positive and below 50%')
    heads=len(table.errors[0]);pairs=len(table.errors)
    flat=PairTable([e for row in table.errors for e in row],table.payload_bits,
                   table.dense_pair_bits,table.tail_bits)
    if offline:
        result=run_offline(flat,target_saving)
        result['history']=[{'tokens':2*(i+1),'formats':[r['format'] for r in result['history'][i*heads:(i+1)*heads]]}
                           for i in range(pairs)]
        return result
    allowed=list(range(5)) if fixed_residual is None else [0,RESIDUALS.index(fixed_residual)+1]
    warmup=min(control.startup_tokens//2,pairs)
    price=initial_price(flat,flat.errors[:warmup*heads],target_saving,allowed)
    initial=price;anchor=max(price,1e-12);rates=[b/table.dense_pair_bits for b in table.payload_bits]
    target=1-target_saving;alpha=1-math.exp(-1/control.memory_pairs)
    payload=previous=hits=0;ema=target;history=[];total_error=0.;counts={n:0 for n in ['dense','r0','r8','r16','r32']}
    for n,row in enumerate(table.errors,1):
        used=price;choices=[select(e,rates,price,allowed) for e in row]
        payload+=sum(table.payload_bits[i] for i in choices)
        stored=payload+metadata_bits(n*heads,3)
        fraction=stored/(n*heads*table.dense_pair_bits)
        increment=(stored-previous)/(heads*table.dense_pair_bits)
        total_error+=sum(e[i] for e,i in zip(row,choices))/heads
        recent_band,cumulative_band,slack=effective_bands(table,allowed,n,control)
        if n==warmup:ema=fraction
        elif n>warmup:
            ema=(1-alpha)*ema+alpha*increment
            if feedback:
                correction=control.recent_gain*asymmetric_outside(ema-target,recent_band,control.recent_deadband)
                correction+=control.cumulative_gain*asymmetric_outside(fraction-target,cumulative_band,control.cumulative_deadband)
                change=max(-control.max_log_step,min(control.max_log_step,correction))
                new=(price if price>0 else anchor/control.price_span)*math.exp(change)
                price=max(anchor/control.price_span,min(anchor*control.price_span,new));hits+=int(price!=new)
        formats=['dense' if i==0 else f'r{RESIDUALS[i-1]}' for i in choices]
        for name in formats:counts[name]+=1
        history.append(dict(tokens=2*n,formats=formats,price_used=used,price_next=price,stored_bits=stored,
            cumulative_saving=1-fraction,recent_saving=1-ema,pair_relative_error=sum(e[i] for e,i in zip(row,choices))/heads))
        previous=stored
    dense=pairs*heads*table.dense_pair_bits+table.tail_bits;stored+=table.tail_bits
    return dict(target_saving=target_saving,measured_saving=1-stored/dense,mean_relative_squared_error=total_error/pairs,
        initial_price=initial,price_bound_hits=hits,startup_saving=history[warmup-1]['cumulative_saving'],
        formats=counts,stored_bits=stored,dense_bits=dense,history=history)
