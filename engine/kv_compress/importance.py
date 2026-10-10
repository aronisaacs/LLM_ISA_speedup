"""Attention received within the available prefill, never future decode queries."""
import torch


def attention_received(query, keys, block=256):
    """Return [all/even/odd queries, KV heads, tokens], normalized by visibility."""
    if query.shape[0] != 1 or keys.shape[0] != 1 or query.shape[-2:] != keys.shape[-2:]:
        raise ValueError("importance requires matching batch-one prefill Q/K")
    heads, length, dim = query.shape[1:]
    kv_heads = keys.shape[1]
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    query, keys = query.float(), keys.float()
    expanded = keys[0].repeat_interleave(heads // kv_heads, dim=0)
    total = torch.zeros(2, heads, length, device=query.device)
    columns = torch.arange(length, device=query.device)
    for start in range(0, length, block):
        rows = torch.arange(start, min(start + block, length), device=query.device)
        scores = query[0, :, rows] @ expanded.transpose(-1, -2) / dim ** .5
        probs = scores.masked_fill(columns > rows[:, None], float('-inf')).softmax(-1)
        for parity in (0, 1):
            total[parity] += probs[:, rows % 2 == parity].sum(1)
    later = (length - columns).float()
    odd = (length - columns) // 2 + ((length - columns) % 2) * (columns % 2)
    counts = torch.stack((later - odd, odd.float())).clamp_min(1)
    split = (total / counts[:, None]).reshape(2, kv_heads, heads // kv_heads, length).sum(2)
    whole = (total.sum(0) / later).reshape(kv_heads, heads // kv_heads, length).sum(1)
    return torch.cat((whole[None], split))


def capture_queries(model_type):
    """A one-layer handoff from Llama rotary to Cache.update; restored on uninstall."""
    if model_type != 'llama':
        raise ValueError("live token importance currently supports Llama only")
    import transformers.models.llama.modeling_llama as llama
    original = llama.apply_rotary_pos_emb
    pending = {}

    def rotary(*args, **kwargs):
        query, key = original(*args, **kwargs)
        pending['query'] = query
        return query, key

    def take():
        return pending.pop('query', None)

    def uninstall():
        llama.apply_rotary_pos_emb = original
        pending.clear()

    llama.apply_rotary_pos_emb = rotary
    return take, uninstall


def next8_attention(query, keys):
    """Bounded local attention: next eight queries, causal 16-token Q/K window.

    Softmax includes the old block and visible tokens of the next block, never
    earlier history or later queries. GQA query heads are averaged per KV head.
    A uniform-attention reference of 1/16 defines weight 1; a fixed 0.1 floor
    protects tokens with little observed local attention. No sequence-wide
    normalization (which would leak later blocks). The final tail is untouched.
    """
    if query.ndim != 4 or keys.ndim != 4 or query.shape[0] != 1 or keys.shape[0] != 1 or query.shape[-2:] != keys.shape[-2:]:
        raise ValueError('next8 importance requires matching batch-one Q/K')
    hq, length, dim = query.shape[1:]
    hk = keys.shape[1]
    if hq % hk or not torch.isfinite(query).all() or not torch.isfinite(keys).all():
        raise ValueError('invalid Q/K heads or nonfinite values')
    weights = torch.ones(1, hk, length, device=keys.device)
    windows = max(0, (length-8)//8)
    if not windows:
        return weights
    # Overlapping local windows; constant 8x16 attention work per block.
    k = keys.float().unfold(2,16,8).permute(0,1,2,4,3)
    q = query.float()[...,8:,:].unfold(2,8,8).permute(0,1,2,4,3)
    q = q[:,:,:windows].reshape(1,hk,hq//hk,windows,8,dim)
    scores = torch.einsum('bhgwqd,bhwkd->bhgwqk',q,k[:,:,:windows])/dim**.5
    causal = torch.arange(16,device=keys.device)[None,:] > torch.arange(8,16,device=keys.device)[:,None]
    probs = scores.masked_fill(causal,float('-inf')).softmax(-1)
    received = (16*probs[...,:8].mean((2,4))).clamp_min(.1)
    weights[...,:windows*8] = received.reshape(1,hk,windows*8)
    return weights
