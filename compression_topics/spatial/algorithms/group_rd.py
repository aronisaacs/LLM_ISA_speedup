"""Rate-distortion choice per 4-token K or V block from a small menu of fixed formats.

Every aligned block of four tokens (per KV head) is stored in one mode, and
every mode has one fixed size. Each of the block's two pairs is dense or
merged with one shared signed residual of r entries, or the whole block is
one quad with r entries per token; r is only ever a value of ``residuals``
(default 0, 4, 8, 16, 32). Mode names: ``D`` (dense block), ``P<a>+<b>``
(pairs, each ``d`` for dense or a residual size, e.g. ``Pd+32``), ``Q<r>``
(quad). These residuals allow (1 + 5)^2 + 5 = 41 modes; ``menu`` picks the
ones a format offers (``D`` is required, ``None`` keeps all of them), and the
mode code takes ceil(log2(modes)) bits. ``DEFAULT_MENU`` has 8 modes, a 3-bit
code, spread evenly over the savings a mode can reach (bitmap masks):
  D 0%, Pd+32 17%, P32+32 34%, P8+8 43%, P0+0 49%, Q16 55%, Q8 62%, Q0 74%.
``modes`` ('pairs', 'quads') restricts the full set before ``menu``.
A menu may also list ``P<a>/<b>`` (a != b): one format covering ``P<a>+<b>``
and ``P<b>+<a>``, the same size, plus one flag bit per block saying which pair
takes which side, so every block can choose its own orientation even when one
format covers a whole page (e.g. ``P32/d``: either pair merged with 32 entries).
A menu may also list ``S<r>``: a quad whose four tokens share one residual
mask, the r features with the largest squared deviation summed over the
block. It stores one position mask instead of four, which pays off when
stored values are narrow (quantized caches).

Residual entries are the largest deviations (``select_by='deviation'``), or for
keys the largest deviations weighted by query energy (``select_by='query'``,
needs ``query_weights``): the entries that move attention scores most.

Each decision unit takes the mode with the smallest distortion + lam * bits,
summed over its blocks. The unit is one block, or ``granularity`` tokens
(16 or 64) of one head, which all share one mode and one code. One price
``lam`` per slot sets the budget: given ``saving`` it is bisected to the
smallest price whose bits reach that saving; given ``lam`` it is used as is.
A fixed price decides every unit from that unit alone, so it is causal.

Pairs and quads reconstruct exactly as ``group_calibrated`` (direction mean,
largest-|d| residual, exact 16-bit norms; keys are first moved into the RoPE
frame of the group's first token, values are used as stored). Distortion per
token, summed over the block:
  cosine   1 - cos(reconstruction, vector)
  squared  ||restored - vector||^2  (= 2 |x|^2 (1 - cos))
  query    keys only: sum_j w_j (restored - key)_j^2, with w the per-layer,
           per-KV-head mean squared query coordinate from a calibration file.
           Weights are averaged over each RoPE plane (j, j + D/2), which makes
           the error the same in every RoPE frame.
For values ``squared`` is the natural choice: attention output is a
probability-weighted sum of values, so its error scales with each value's
absolute error, which ``cosine`` ignores. Each (target, layer) slot has its
own price and budget, so K and V slots can share one budget pool.
Bits count merged vectors, residual values, residual masks (``bitmap``: a
D-bit mask per residual; ``index``: ceil(log2 D) bits per entry; ``auto``:
the smaller of the two for each residual size; each is one fixed size per
mode), 16-bit norms, the byte-packed mode codes of all units and a two-byte
slot header.

Decode: prefill compresses whole units only and keeps the price it used per
slot. Each later cache update compresses the units it completes at that
price (or at ``lam``); tokens of an unfinished unit stay dense until it
completes. Since a unit's mode depends only on that unit, prefill plus decode
makes the same choices as compressing the whole sequence at that price.
``decode=False`` keeps generated tokens dense.
Pairs-only or quads-only with one residual size match the
``group_calibrated`` format bit for bit. Batch size one.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from compression_topics.spatial.algorithms.storage import metadata_bits, residual_mask_bits, vector_group_bits
from engine.kv_compress.methods import AFTER_APPEND, METHODS
from engine.kv_compress.metrics import STATS
from engine.kv_compress.rope import apply_rope

ACCOUNTING = 'fp16_rd_block_modes_norms_header_v1'
RESIDUALS = (0, 4, 8, 16, 32)
DEFAULT_MENU = ('D', 'Pd+32', 'P32+32', 'P8+8', 'P0+0', 'Q16', 'Q8', 'Q0')
MODES = ('both', 'pairs', 'quads')
DISTORTIONS = ('cosine', 'squared', 'query')
MASKS = ('bitmap', 'index', 'auto')
GRANULARITIES = (4, 16, 64)
BLOCK = 4
ROOT = Path(__file__).resolve().parents[3]
# Price each layer's prefill used, for the units that decode completes.
PRICES = {}


def mode_names(residuals=RESIDUALS, modes='both', menu=None):
    """Menu in code order. Pair choice 0 is a dense pair, i + 1 is residual i."""
    labels = ['d'] + [str(r) for r in residuals]
    names = []
    if modes in ('both', 'pairs'):
        names += ['D' if a == b == 0 else f'P{labels[a]}+{labels[b]}'
                  for a in range(len(labels)) for b in range(len(labels))]
    else:
        names.append('D')
    if modes in ('both', 'quads'):
        names += [f'Q{r}' for r in residuals]
    if menu is not None:
        extra = [name for name in menu if '/' in name or name[:1] == 'S']
        valid = {f'P{a}/{b}' for a in labels for b in labels if a != b} if modes in ('both', 'pairs') else set()
        valid |= {f'S{r}' for r in residuals} if modes in ('both', 'quads') else set()
        unknown = (set(menu) - set(names) - valid) | (set(extra) - valid)
        if unknown or 'D' not in menu or len(set(menu)) != len(menu):
            raise ValueError(f"menu must list distinct modes including 'D'; unknown: {sorted(unknown)}")
        names = [name for name in names if name in set(menu)] + extra
    return names


def mode_bits(count):
    return max(1, math.ceil(math.log2(count)))


def option_bits(residuals=RESIDUALS, dim=128, mask='bitmap', value_bits=16):
    """Bits of a merged pair and of a quad (per-token masks, then shared mask) for each residual size.

    ``value_bits`` is the width of stored vector entries (16 in the simulation; smaller values only
    model a quantized cache's storage); norms stay 16-bit.
    """
    pair = [value_bits * (dim + r) + 16 * 2 + residual_mask_bits(r, dim, 1, mask) for r in residuals]
    quad = [value_bits * (dim + BLOCK * r) + 16 * BLOCK + residual_mask_bits(r, dim, BLOCK, mask) for r in residuals]
    shared = [value_bits * (dim + BLOCK * r) + 16 * BLOCK + residual_mask_bits(r, dim, 1, mask) for r in residuals]
    return pair, quad + shared


def _parse(name, residuals):
    """(is_quad, first pair choice, second pair choice, quad index) of a mode name.

    Quad indices run over per-token-mask quads, then shared-mask quads.
    """
    index = {str(r): i for i, r in enumerate(residuals)}
    if name == 'D':
        return False, 0, 0, 0
    if name[0] == 'Q':
        return True, 0, 0, index[name[1:]]
    if name[0] == 'S':
        return True, 0, 0, len(residuals) + index[name[1:]]
    first, second = name[1:].replace('/', '+').split('+')
    choice = lambda label: 0 if label == 'd' else index[label] + 1
    return False, choice(first), choice(second), 0


def mode_columns(names, residuals=RESIDUALS, device=None):
    """(is_quad, first pair choice, second pair choice, quad index) tensors over modes."""
    parsed = [_parse(name, residuals) for name in names]
    return tuple(torch.tensor(column, device=device) for column in zip(*parsed))


def mode_flags(names, device=None):
    """Modes whose blocks carry a one-bit pair-orientation flag."""
    return torch.tensor(['/' in name for name in names], device=device)


def bit_table(names, residuals=RESIDUALS, dim=128, mask='bitmap', device=None, value_bits=16):
    """Bits of one block in each mode, before mode codes and the slot header."""
    is_quad, first, second, quad = mode_columns(names, residuals, device)
    pair_bits, quad_bits = option_bits(residuals, dim, mask, value_bits)
    pair_table = torch.tensor([2 * dim * value_bits] + pair_bits, dtype=torch.float64, device=device)
    quad_table = torch.tensor(quad_bits, dtype=torch.float64, device=device)
    flags = mode_flags(names, device).double()
    return torch.where(is_quad, quad_table[quad], pair_table[first] + pair_table[second] + flags)


@functools.lru_cache(maxsize=4)
def _load_weights(path):
    file = Path(path)
    if not file.is_absolute():
        file = ROOT / file
    payload = torch.load(file, map_location='cpu', weights_only=True)
    return payload['weights'].float()  # [layers, kv_heads, head_dim]


def plane_tied(weights):
    """Average each RoPE plane (j, j + D/2) so the error ignores RoPE angles."""
    half = weights.shape[-1] // 2
    mean = (weights[..., :half] + weights[..., half:]) / 2
    return torch.cat((mean, mean), dim=-1)


@dataclass
class Menu:
    """Distortion of every mode for every block of one slot."""
    names: list
    distortion: torch.Tensor  # [1, H, blocks, modes], float64
    bits: torch.Tensor        # [modes], float64
    is_quad: torch.Tensor     # [modes] bool
    first: torch.Tensor       # [modes] pair choices
    second: torch.Tensor
    quad: torch.Tensor        # [modes] quad residual index
    swap: torch.Tensor        # [1, H, blocks, modes] bool: flagged pair modes take the swapped orientation
    pair_recon: list          # per residual: aligned unit directions [1, H, blocks, 2, 2, D]
    quad_recon: list          # per residual: aligned unit directions [1, H, blocks, 4, D]


def _distortion(recon, unit, norms, kind, weights):
    """Per-token error of the reconstruction ``recon`` against ``unit``."""
    length = recon.norm(dim=-1, keepdim=True)
    direction = recon / length.clamp_min(1e-6)
    if kind == 'cosine':
        error = 1 - (direction * unit).sum(-1)
    else:
        delta = (direction - unit) * norms
        error = (delta.square() * weights).sum(-1) if kind == 'query' else delta.square().sum(-1)
    invalid = (length.squeeze(-1) <= 1e-6) | (norms.squeeze(-1) <= 1e-12)
    return error.masked_fill(invalid, float('inf')), direction


def _tables(length, group, rope_tables, device):
    """RoPE tables that move each token into its group's first-token frame."""
    return rope_tables.cos_sin(torch.arange(length, device=device) % group, torch.float32)


def _rotate(vectors, group, rope_tables, inverse):
    """Into (``inverse``) or out of the group's first-token RoPE frame; values pass ``None``."""
    if rope_tables is None:
        return vectors
    cos, sin = _tables(vectors.shape[-2], group, rope_tables, vectors.device)
    return apply_rope(vectors, cos, sin, inverse=inverse)


def _aligned(original, group, rope_tables):
    return _rotate(original, group, rope_tables, inverse=True)


def build_menu(tensor, *, rope_tables, residuals=RESIDUALS, modes='both', menu=None,
               mask='bitmap', distortion='cosine', weights=None, select_by='deviation',
               token_weights=None):
    """Distortion of every mode; ``tensor`` is [1, H, T, D], T a multiple of 4.

    Keys pass ``rope_tables`` (aligned to each group's first token); values pass ``None``.
    ``token_weights`` [1, H, T] scales each token's error (e.g. the attention it receives).
    """
    _, heads, length, dim = tensor.shape
    blocks = length // BLOCK
    original = tensor.float()
    norms = original.norm(dim=-1, keepdim=True)
    tied = plane_tied(weights.float()).to(original.device) if weights is not None else None
    weights = tied if distortion == 'query' else None
    rank = (lambda deviation, shape: deviation.square() * tied.reshape(shape)) if select_by == 'query' else \
        (lambda deviation, shape: deviation.square())
    top = max(residuals)
    # Pairs: one half-difference and one mask; the second residual is its negative.
    unit = torch.nn.functional.normalize(_aligned(original, 2, rope_tables), dim=-1)
    unit = unit.reshape(1, heads, blocks, 2, 2, dim)
    scale = norms.reshape(1, heads, blocks, 2, 2, 1)
    mean = unit.mean(-2)
    delta = (unit[..., 0, :] - unit[..., 1, :]) / 2
    index = rank(delta, (1, heads, 1, 1, dim)).topk(top, dim=-1).indices if top else None
    pair_d, pair_recon = [], []
    for r in residuals:
        residual = torch.zeros_like(delta)
        if r:
            residual.scatter_(-1, index[..., :r], delta.gather(-1, index[..., :r]))
        error, direction = _distortion(torch.stack((mean + residual, mean - residual), dim=-2), unit, scale,
                                       distortion, None if weights is None else weights.reshape(1, heads, 1, 1, 1, dim))
        if token_weights is not None:
            error = error * token_weights.clamp_min(1e-12).reshape(1, heads, blocks, 2, 2)
        pair_d.append(error.sum(-1))
        pair_recon.append(direction)
    # Quads: one mean and a separate residual per token.
    unit = torch.nn.functional.normalize(_aligned(original, BLOCK, rope_tables), dim=-1)
    unit = unit.reshape(1, heads, blocks, BLOCK, dim)
    scale = norms.reshape(1, heads, blocks, BLOCK, 1)
    mean = unit.mean(-2, keepdim=True)
    spread = unit - mean
    score = rank(spread, (1, heads, 1, 1, dim))
    own = score.topk(top, dim=-1).indices if top else None
    shared = score.sum(-2, keepdim=True).topk(top, dim=-1).indices.expand(*spread.shape[:-1], top) if top else None
    quad_d, quad_recon = [], []
    for index in (own, shared):  # per-token masks, then one shared mask per block
        for r in residuals:
            residual = torch.zeros_like(spread)
            if r:
                residual.scatter_(-1, index[..., :r], spread.gather(-1, index[..., :r]))
            error, direction = _distortion(mean + residual, unit, scale, distortion,
                                           None if weights is None else weights.reshape(1, heads, 1, 1, dim))
            if token_weights is not None:
                error = error * token_weights.clamp_min(1e-12).reshape(1, heads, blocks, BLOCK)
            quad_d.append(error.sum(-1))
            quad_recon.append(direction)
    names = mode_names(residuals, modes, menu)
    device = tensor.device
    is_quad, first, second, quad = mode_columns(names, residuals, device)
    bits = bit_table(names, residuals, dim, mask, device)
    pairs = torch.cat((torch.zeros(1, heads, blocks, 2, 1, device=device), torch.stack(pair_d, -1)), -1).double()
    quads = torch.stack(quad_d, -1).double()
    straight = pairs[..., 0, first] + pairs[..., 1, second]
    swapped = pairs[..., 0, second] + pairs[..., 1, first]
    swap = mode_flags(names, device) & (swapped < straight)
    table = torch.where(is_quad, quads[..., quad], torch.where(swap, swapped, straight))
    return Menu(names, table, bits, is_quad, first, second, quad, swap, pair_recon, quad_recon)


def block_modes(menu, choice):
    """Per block: (is_quad, first pair choice, second pair choice, quad index), flags applied."""
    first, second = menu.first[choice], menu.second[choice]
    swap = menu.swap.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    return (menu.is_quad[choice], torch.where(swap, second, first), torch.where(swap, first, second),
            menu.quad[choice])


def _units(values, per_unit):
    """Sum [..., blocks, modes] over consecutive runs of ``per_unit`` blocks."""
    blocks = values.shape[-2]
    padded = math.ceil(blocks / per_unit) * per_unit
    if padded != blocks:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded - blocks))
    return values.reshape(*values.shape[:-2], padded // per_unit, per_unit, values.shape[-1]).sum(-2)


def select(menu, lam, granularity=BLOCK):
    """Mode index per block; ``lam=None`` minimizes bits over valid modes."""
    bits = menu.bits.expand_as(menu.distortion)
    if lam is None:
        cost = bits + torch.where(torch.isinf(menu.distortion), float('inf'), 0.)
    else:
        cost = menu.distortion + lam * bits
    per_unit = granularity // BLOCK
    choice = _units(cost, per_unit).argmin(-1)
    return choice.repeat_interleave(per_unit, dim=-1)[..., :menu.distortion.shape[-2]]


def units(menu, granularity=BLOCK):
    """Decision units (one mode code each) of all heads."""
    heads, blocks = menu.distortion.shape[1:3]
    return heads * math.ceil(blocks / (granularity // BLOCK))


def total_bits(menu, lam, granularity=BLOCK):
    return float(menu.bits[select(menu, lam, granularity)].sum())


def solve_lambda(menu, target, granularity=BLOCK, iterations=60):
    """Smallest price (within bisection precision) whose block bits are at most ``target``."""
    if total_bits(menu, 0., granularity) <= target:
        return 0.
    if total_bits(menu, None, granularity) > target:
        raise ValueError("saving exceeds the metadata-inclusive representation maximum")
    lo, hi = 0., 1e-12
    while total_bits(menu, hi, granularity) > target:
        lo, hi = hi, hi * 4
    for _ in range(iterations):
        middle = math.sqrt(lo * hi) if lo > 0 else hi / 2
        if total_bits(menu, middle, granularity) <= target:
            hi = middle
        else:
            lo = middle
    return hi


def reconstruct(tensor, menu, choice, rope_tables):
    """Every block replaced by its chosen mode; dense pairs stay exact."""
    _, heads, length, dim = tensor.shape
    blocks = length // BLOCK
    original = tensor.float()
    norms = original.norm(dim=-1, keepdim=True)
    is_quad, first, second, quad = block_modes(menu, choice)
    pair_choice = torch.stack((first, second), -1).masked_fill(is_quad.unsqueeze(-1), 0)
    chosen = torch.zeros(1, heads, blocks, 2, 2, dim, device=tensor.device)
    for index, direction in enumerate(menu.pair_recon):
        chosen = torch.where((pair_choice == index + 1)[..., None, None], direction, chosen)
    restored = _rotate(chosen.reshape(1, heads, length, dim) * norms, 2, rope_tables, inverse=False)
    merged = (pair_choice > 0).unsqueeze(-1).expand(*pair_choice.shape, 2).reshape(1, heads, length, 1)
    out = torch.where(merged, restored, original)
    chosen = torch.zeros(1, heads, blocks, BLOCK, dim, device=tensor.device)
    for index, direction in enumerate(menu.quad_recon):
        chosen = torch.where((is_quad & (quad == index))[..., None, None], direction, chosen)
    restored = _rotate(chosen.reshape(1, heads, length, dim) * norms, BLOCK, rope_tables, inverse=False)
    quads = is_quad.unsqueeze(-1).expand(*is_quad.shape, BLOCK).reshape(1, heads, length, 1)
    return torch.where(quads, restored, out)


def _check(target, residuals, modes, mask, distortion, query_weights, saving, lam,
           granularity, accounting, select_by='deviation'):
    if accounting != ACCOUNTING:
        raise ValueError("unsupported rate-distortion storage accounting version")
    if target not in ('k', 'v'):
        raise ValueError("target must be k or v")
    if target == 'v' and distortion == 'query':
        raise ValueError("values have no query weighting; use 'squared' or 'cosine'")
    if (saving is None) == (lam is None):
        raise ValueError("give exactly one of saving or lam")
    for value, name in ((saving, 'saving'), (lam, 'lam')):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number")
    if saving is not None and saving >= 1:
        raise ValueError("saving must be below one")
    if (not residuals or len(set(residuals)) != len(residuals)
            or any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in residuals)):
        raise ValueError("residuals must be distinct non-negative integers")
    if modes not in MODES or mask not in MASKS or distortion not in DISTORTIONS:
        raise ValueError(f"modes in {MODES}, mask in {MASKS}, distortion in {DISTORTIONS}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES} tokens")
    if select_by not in ('deviation', 'query'):
        raise ValueError("select_by must be 'deviation' or 'query'")
    if target == 'v' and select_by == 'query':
        raise ValueError("values have no query weighting; use select_by='deviation'")
    if ('query' in (distortion, select_by)) != (query_weights is not None):
        raise ValueError("distortion or select_by 'query' needs query_weights, and only they use them")


def apply(tensor, *, layer_idx, target, saving=None, lam=None, residuals=RESIDUALS,
          modes='both', menu=DEFAULT_MENU, granularity=BLOCK, mask='bitmap', distortion='cosine',
          query_weights=None, select_by='deviation', decode=True, seq_start=0, rope_tables=None,
          accounting=ACCOUNTING,
          **unused):
    residuals = tuple(residuals)
    _check(target, residuals, modes, mask, distortion, query_weights, saving, lam,
           granularity, accounting, select_by)
    names = mode_names(residuals, modes, menu)
    if seq_start != 0 or saving == 0 or lam == 0:
        return tensor
    rope_tables = _check_tensor(tensor, target, rope_tables, residuals)
    full = tensor.shape[-2] // granularity * granularity
    PRICES.pop((target, layer_idx), None)
    if not full:
        return tensor
    body = tensor[..., :full, :]
    table = build_menu(body, rope_tables=rope_tables, residuals=residuals, modes=modes, menu=menu,
                       mask=mask, distortion=distortion, weights=_weights(query_weights, layer_idx),
                       select_by=select_by)
    overhead = metadata_bits(units(table, granularity), mode_bits(len(names)))
    dense_bits = tensor.numel() * 16
    tail_bits = (tensor.shape[-2] - full) * tensor.shape[1] * tensor.shape[-1] * 16
    if saving is not None:
        lam = solve_lambda(table, dense_bits * (1 - saving) - overhead - tail_bits, granularity)
    PRICES[target, layer_idx] = lam
    choice = select(table, lam, granularity)
    out = tensor.clone()
    out[..., :full, :] = reconstruct(body, table, choice, rope_tables).to(tensor.dtype)
    _record(target, layer_idx, body, out[..., :full, :], table, choice, dense_bits=dense_bits,
            stored_bits=float(table.bits[choice].sum()) + overhead + tail_bits, overhead=overhead,
            lam=lam, residuals=residuals, mask=mask, granularity=granularity)
    return out


def after_append(tensor, *, target, layer_idx, start, end, rope_tables=None, saving=None, lam=None,
                 residuals=RESIDUALS, modes='both', menu=DEFAULT_MENU, granularity=BLOCK, mask='bitmap',
                 distortion='cosine', query_weights=None, select_by='deviation', decode=True,
                 accounting=ACCOUNTING, **unused):
    """Compress the units that the tokens ``[start, end)`` of the stored tensor complete."""
    if start == 0 or not decode or saving == 0 or lam == 0:
        return  # Prefill (start 0) is compressed by ``apply``.
    residuals = tuple(residuals)
    _check(target, residuals, modes, mask, distortion, query_weights, saving, lam,
           granularity, accounting, select_by)
    rope_tables = _check_tensor(tensor, target, rope_tables, residuals)
    if lam is None:
        if (target, layer_idx) not in PRICES:
            raise ValueError("decode needs the price of this slot's prefill, or a fixed lam")
        lam = PRICES[target, layer_idx]
    names = mode_names(residuals, modes, menu)
    new_bits = (end - start) * tensor.shape[1] * tensor.shape[-1] * 16
    first, last = start // granularity * granularity, end // granularity * granularity
    if last <= first:
        _record(target, layer_idx, None, None, None, None, dense_bits=new_bits, stored_bits=new_bits,
                overhead=0, lam=lam, residuals=residuals, mask=mask, granularity=granularity)
        return
    body = tensor[..., first:last, :]
    table = build_menu(body, rope_tables=rope_tables, residuals=residuals, modes=modes, menu=menu,
                       mask=mask, distortion=distortion, weights=_weights(query_weights, layer_idx),
                       select_by=select_by)
    choice = select(table, lam, granularity)
    restored = reconstruct(body, table, choice, rope_tables).to(tensor.dtype)
    tensor[..., first:last, :] = restored
    overhead = units(table, granularity) * mode_bits(len(names))  # appended to the packed codes
    stored = float(table.bits[choice].sum()) + overhead + new_bits - body.numel() * 16
    _record(target, layer_idx, body, restored, table, choice, dense_bits=new_bits, stored_bits=stored,
            overhead=overhead, lam=lam, residuals=residuals, mask=mask, granularity=granularity)


def _weights(query_weights, layer_idx):
    return None if query_weights is None else _load_weights(str(query_weights))[layer_idx]


def _check_tensor(tensor, target, rope_tables, residuals):
    """Validate the tensor; returns the RoPE tables to align with (none for values)."""
    if tensor.shape[0] != 1:
        raise ValueError("group_rd requires batch size one")
    if max(residuals) > tensor.shape[-1]:
        raise ValueError("residuals must fit the head dimension")
    if target == 'v':
        return None
    if rope_tables is None or rope_tables.attention_scaling != 1.:
        raise ValueError("keys require pure-rotation RoPE tables")
    return rope_tables


def _record(target, layer_idx, body, restored, menu, choice, *, dense_bits, stored_bits, overhead, lam,
            residuals, mask, granularity):
    key = f'{target}_layer_{layer_idx}'
    row = STATS.setdefault(key, {'dense_bits': 0, 'stored_bits': 0, 'pairs': 0, 'merged': 0,
        'features': None, 'accounting': ACCOUNTING, 'mask': mask, 'granularity': granularity,
        'modes': None, 'metadata_bits': 0, 'norm_bits': 0, 'residual_mask_bits': 0,
        'cosine_sum': 0., 'cosine_min': 1., 'cutoff_sum': 0., 'updates': 0,
        'shortfall_updates': 0, 'lambda_sum': 0., 'distortion_sum': 0.,
        'dense_blocks': 0, 'quad_blocks': 0, 'merged_pairs': 0,
        **{f'pair_r{r}': 0 for r in residuals}, **{f'quad_r{r}': 0 for r in residuals},
        **{f'shared_r{r}': 0 for r in residuals}})
    row['dense_bits'] += dense_bits
    row['stored_bits'] += int(stored_bits)
    if menu is None:
        return
    dim = body.shape[-1]
    row['features'], row['modes'] = dim, len(menu.names)
    is_quad, first, second, quad = block_modes(menu, choice)
    pair_choice = torch.stack((first, second), -1).masked_fill(is_quad.unsqueeze(-1), 0)
    merged = is_quad | (pair_choice > 0).any(-1)
    cosine = torch.nn.functional.cosine_similarity(restored.float(), body.float(), dim=-1)
    block_min = cosine.reshape(merged.shape + (BLOCK,)).amin(-1)
    pair_counts = [int((pair_choice == index + 1).sum()) for index in range(len(residuals))]
    quad_counts = [int((is_quad & (quad == index)).sum()) for index in range(len(residuals))]
    shared_counts = [int((is_quad & (quad == len(residuals) + index)).sum()) for index in range(len(residuals))]
    row['metadata_bits'] += overhead
    row['norm_bits'] += 16 * (2 * sum(pair_counts) + BLOCK * (sum(quad_counts) + sum(shared_counts)))
    row['residual_mask_bits'] += sum(p * residual_mask_bits(r, dim, 1, mask) + q * residual_mask_bits(r, dim, BLOCK, mask)
                                     + c * residual_mask_bits(r, dim, 1, mask)
                                     for r, p, q, c in zip(residuals, pair_counts, quad_counts, shared_counts))
    row['pairs'] += merged.numel()  # Existing measurement schema; these are blocks.
    row['merged'] += int(merged.sum())
    row['updates'] += 1
    row['lambda_sum'] += float(lam)
    row['distortion_sum'] += float(menu.distortion.gather(-1, choice.unsqueeze(-1)).sum())
    row['dense_blocks'] += int((~merged).sum())
    row['quad_blocks'] += sum(quad_counts) + sum(shared_counts)
    row['merged_pairs'] += sum(pair_counts)
    for r, p, q, c in zip(residuals, pair_counts, quad_counts, shared_counts):
        row[f'pair_r{r}'] += p
        row[f'quad_r{r}'] += q
        row[f'shared_r{r}'] += c
    if merged.any():
        chosen = block_min[merged]
        row['cosine_sum'] += float(chosen.sum())
        row['cosine_min'] = min(row['cosine_min'], float(chosen.min()))
        row['cutoff_sum'] += float(chosen.min())


METHODS['group_rd'] = apply
AFTER_APPEND['group_rd'] = after_append
