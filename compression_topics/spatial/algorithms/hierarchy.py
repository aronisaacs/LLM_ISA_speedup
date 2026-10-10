"""Bounded pair/quad/eight tree decisions inside consecutive eight-token windows.

One format choice spans KV heads; every mean/residual remains head-local.
Leaves are dense pairs or merged pairs. A quad/eight may replace its children.
There are at most eight terminal formats, not an enumeration of partitions.
Prefill prototype only: decode appends remain dense. Byte counts model a packed
stream of terminal tags, BF16 means/scales/residuals, per-token masks for g>2,
and a 16-bit slot header. No allocation or accuracy claim is made by this kernel.
"""
from dataclasses import dataclass
import math
import torch

from compression_topics.spatial.algorithms.decisions import Control, asymmetric_outside
from compression_topics.spatial.algorithms.representations import prepare_groups, reconstruct
from compression_topics.spatial.algorithms.scoring import reconstruction_error
from compression_topics.spatial.algorithms.storage import metadata_bits, vector_group_bits
from engine.kv_compress.rope import apply_rope

# Nested subsets keep pair residuals identical in the three pilot variants.
MENUS = {2: (0, 16, 32), 4: (0, 16), 8: (0, 16)}
WINDOW = 8
TAG_BITS = 3
ACCOUNTING = 'hierarchical_bf16_scales_bitmap_terminal_3bit_v1'


def accounting(mask_layout):
    if mask_layout not in ('separate', 'shared'):
        raise ValueError('mask_layout must be separate or shared')
    return ACCOUNTING if mask_layout == 'separate' else 'hierarchical_bf16_scales_shared_bitmap_terminal_3bit_v1'


@dataclass(frozen=True)
class Choice:
    segments: tuple  # (window-relative start, size, residual count; None = dense pair)
    error: float  # relative error, token-weighted to one eight-token window
    payload_bits: int

    @property
    def bits(self):
        return self.payload_bits + TAG_BITS * len(self.segments)


@dataclass
class WindowTable:
    rows: list  # each dict maps (size, relative start) to errors for that size's menu
    payload: dict
    dense_window_bits: int
    tail_bits: int
    max_group: int
    mask_layout: str = 'separate'

    def choose(self, row, price):
        """Bottom-up dynamic program, O(nodes x short residual menu)."""
        def visit(start, size):
            if size == 2:
                best = Choice(((start, 2, None),), 0., self.payload[(2, None)])
            else:
                left, right = visit(start, size // 2), visit(start + size // 2, size // 2)
                best = Choice(left.segments + right.segments, left.error + right.error,
                              left.payload_bits + right.payload_bits)
            cost = best.error + price * best.bits / self.dense_window_bits
            if size <= self.max_group:
                for residual, error in zip(MENUS[size], row[(size, start)]):
                    option = Choice(((start, size, residual),), error, self.payload[(size, residual)])
                    candidate = option.error + price * option.bits / self.dense_window_bits
                    # Retain children at a tie, rather than force a larger merge.
                    if candidate < cost:
                        best, cost = option, candidate
            return best
        return visit(0, WINDOW)


def build_table(tensor, *, target, max_group=8, rope_tables=None, return_reconstruction=False, mask_layout='separate',
                selection='deviation', query_energy=None, token_weights=None):
    accounting(mask_layout)
    if max_group not in MENUS:
        raise ValueError('max_group must be 2, 4 or 8')
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] < 32:
        raise ValueError('need batch-one [1, KV heads, tokens, head_dim >= 32]')
    heads, length, dim = tensor.shape[1:]
    full = length // WINDOW * WINDOW
    if not full:
        raise ValueError('need at least one eight-token window')
    body = tensor[..., :full, :]
    rows = [{} for _ in range(full // WINDOW)]
    payload = {(2, None): heads * 2 * dim * 16}
    reconstructions = {}
    for size, menu in MENUS.items():
        if size > max_group:
            continue
        basis = prepare_groups(body, target=target, group_size=size, directional=True, rope_tables=rope_tables)
        errors = []
        for residual in menu:
            restored, valid = reconstruct(basis, residual, dtype=torch.bfloat16,
                                          round_mean_before_residual=True, shared_mask=mask_layout=='shared',
                                          selection=selection, query_energy=query_energy)
            weights = None if token_weights is None else token_weights[..., :full].reshape(1, heads, full//size, size)
            error = reconstruction_error(basis.source, restored, token_weights=weights).mean((0, 1, 3)) * size / WINDOW
            errors.append(error.masked_fill(~valid.all((0, 1)), float('inf')))
            payload[(size, residual)] = heads * vector_group_bits(residual, dim, size, include_norms=True,
                                                                 shared_mask=mask_layout=='shared')
            if return_reconstruction:
                flat = restored.reshape(1, heads, full, dim)
                if target == 'k':
                    flat = apply_rope(flat, basis.cos, basis.sin, inverse=False)
                reconstructions[(size, residual)] = flat.to(tensor.dtype)
        matrix = torch.stack(errors, -1).cpu().tolist()
        for window, row in enumerate(rows):
            for offset in range(0, WINDOW, size):
                row[(size, offset)] = matrix[(window * WINDOW + offset) // size]
    table = WindowTable(rows, payload, heads * WINDOW * dim * 16,
                        (length - full) * heads * dim * 16, max_group, mask_layout)
    return (table, reconstructions) if return_reconstruction else table


def packed_bits(choices):
    return sum(c.payload_bits for c in choices) + metadata_bits(sum(len(c.segments) for c in choices), TAG_BITS)


def initial_price(table, rows, saving):
    """Fit only startup rows, retaining the undercompressed side of a jump."""
    dense = len(rows) * table.dense_window_bits
    limit = dense * (1 - saving)
    def stored(price):
        return packed_bits([table.choose(row, price) for row in rows])
    low, high = 0., 1e-6
    while stored(high) > limit:
        high *= 4
        if high > 1e12:
            raise ValueError('startup target cannot be met with available hierarchy formats')
    for _ in range(60):
        middle = (low + high) / 2
        if stored(middle) <= limit:
            high = middle
        else:
            low = middle
    return high if abs(stored(high) - limit) <= 1e-8 else low


def run(table, saving, *, control=Control(), feedback=True):
    if not math.isfinite(saving) or not 0 < saving < 1:
        raise ValueError('saving must be between zero and one')
    if not table.rows:
        raise ValueError('need windows')
    if control.startup_tokens % WINDOW:
        raise ValueError('hierarchical startup must contain complete eight-token windows')
    warmup = min(control.startup_tokens // WINDOW, len(table.rows))
    price = initial_price(table, table.rows[:warmup], saving)
    anchor = max(price, 1e-12)
    initial = price
    target = 1 - saving
    # Keep controller memory measured in tokens, as in the pair method.
    pair_equivalents = WINDOW / 2
    alpha = 1 - math.exp(-pair_equivalents / control.memory_pairs)
    ema = target
    choices, history = [], []
    payload = tags = previous = hits = 0
    for n, row in enumerate(table.rows, 1):
        used_price = price
        choice = table.choose(row, price)  # Never inspects a later window.
        choices.append(choice)
        payload += choice.payload_bits
        tags += len(choice.segments)
        stored = payload + metadata_bits(tags, TAG_BITS)
        fraction = stored / (n * table.dense_window_bits)
        increment = (stored - previous) / table.dense_window_bits
        dense_choice = table.choose(row, 0.)
        smallest = table.choose(row, 1e12)
        slack = (dense_choice.bits - smallest.bits) / (n * table.dense_window_bits)
        if n == warmup:
            ema = fraction
        elif n > warmup:
            ema = (1 - alpha) * ema + alpha * increment
            if feedback:
                correction = control.recent_gain * asymmetric_outside(
                    ema - target, control.recent_deadband + slack, control.recent_deadband)
                correction += control.cumulative_gain * asymmetric_outside(
                    fraction - target, control.cumulative_deadband + slack, control.cumulative_deadband)
                # Updates occur once per eight tokens, rather than once per pair.
                step = pair_equivalents * max(-control.max_log_step, min(control.max_log_step, correction))
                new = (price if price > 0 else anchor / control.price_span) * math.exp(step)
                price = max(anchor / control.price_span, min(anchor * control.price_span, new))
                hits += int(price != new)
        history.append(dict(tokens=n * WINDOW, price_used=used_price, price_next=price,
                            cumulative_saving=1-fraction, recent_saving=1-ema,
                            segments=[list(s) for s in choice.segments]))
        previous = stored
    dense = len(table.rows) * table.dense_window_bits + table.tail_bits
    stored += table.tail_bits
    formats = {}
    for choice in choices:
        for _, size, residual in choice.segments:
            name = 'dense_pair' if residual is None else f'g{size}_r{residual}'
            formats[name] = formats.get(name, 0) + 1
    return dict(target_saving=saving, mask_layout=table.mask_layout, measured_saving=1-stored/dense,
                dense_bits=dense, stored_bits=stored, metadata_bits=metadata_bits(tags, TAG_BITS),
                mean_relative_squared_error=sum(c.error for c in choices)/len(choices),
                initial_price=initial, price_bound_hits=hits, formats=formats,
                startup_saving=history[warmup-1]['cumulative_saving'], history=history)


def restore(tensor, result, reconstructions):
    out = tensor.clone()
    full = len(result['history'])*WINDOW
    options = list(reconstructions)
    codes = [0]*full
    for window, item in enumerate(result['history']):
        for start, size, residual in item['segments']:
            if residual is not None:
                offset = window * WINDOW + start
                codes[offset:offset+size] = [options.index((size,residual))+1]*size
    codes = torch.tensor(codes,device=tensor.device).reshape(1,1,full,1)
    body = out[...,:full,:]
    for code, option in enumerate(options,1):
        body = torch.where(codes==code,reconstructions[option],body)
    out[...,:full,:] = body
    return out


def apply(tensor, *, layer_idx, target, saving, max_group=8, control=None, mask_layout='separate',
          seq_start=0, rope_tables=None, selection='deviation', query_energy=None,
          token_weights=None, delayed=False):
    """Head-local hierarchy integrated with prefill; decode appends stay dense."""
    from engine.kv_compress.metrics import STATS
    storage_layout = accounting(mask_layout)
    if max_group not in MENUS:
        raise ValueError('max_group must be 2, 4 or 8')
    settings = Control(**(control or {}))
    if seq_start or tensor.shape[-2] < WINDOW:
        # Report unchanged short prefills; cache accounting includes them.
        if not seq_start:
            key = f'{target}_layer_{layer_idx}'
            row = STATS.setdefault(key, {'accounting': storage_layout, 'dense_bits': 0, 'stored_bits': 0})
            row['dense_bits'] += tensor.numel()*16
            row['stored_bits'] += tensor.numel()*16
        return tensor
    # Only complete windows with eight subsequent observed tokens are eligible.
    eligible = max(0, (tensor.shape[-2]-8)//8*8) if delayed else tensor.shape[-2]
    if delayed and eligible < WINDOW:
        key = f'{target}_layer_{layer_idx}'
        row = STATS.setdefault(key, {'accounting': storage_layout, 'dense_bits': 0, 'stored_bits': 0})
        row['dense_bits'] += tensor.numel()*16
        row['stored_bits'] += tensor.numel()*16
        return tensor
    body = tensor[..., :eligible, :]
    table, reconstructions = build_table(body, target=target, max_group=max_group,
                                         rope_tables=rope_tables, return_reconstruction=True, mask_layout=mask_layout,
                                         selection=selection, query_energy=query_energy, token_weights=token_weights)
    result = run(table, saving, control=settings)
    out = tensor.clone()
    out[..., :eligible, :] = restore(body, result, reconstructions)
    tail_bits = (tensor.shape[-2]-eligible)*tensor.shape[1]*tensor.shape[-1]*16
    result['dense_bits'] += tail_bits
    result['stored_bits'] += tail_bits
    result['measured_saving'] = 1-result['stored_bits']/result['dense_bits']
    key = f'{target}_layer_{layer_idx}'
    row = STATS.setdefault(key, {'accounting': storage_layout, 'dense_bits': 0, 'stored_bits': 0})
    for field in ('dense_bits', 'stored_bits', 'metadata_bits'):
        row[field] = row.get(field, 0) + result[field]
    row['updates'] = row.get('updates', 0) + 1
    row['controller_bound_hits'] = row.get('controller_bound_hits', 0) + result['price_bound_hits']
    row['max_final_target_deviation'] = max(row.get('max_final_target_deviation', 0.), abs(result['measured_saving']-saving))
    for name,count in result['formats'].items():
        row['format_'+name] = row.get('format_'+name,0)+count
    return out
