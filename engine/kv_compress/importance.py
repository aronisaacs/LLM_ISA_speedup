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
