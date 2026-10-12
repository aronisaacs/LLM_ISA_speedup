"""Capture post-RoPE queries next to Cache.update (used to calibrate query energy)."""


def capture_queries(model_type):
    """A one-layer handoff from Llama rotary to Cache.update; restored on uninstall."""
    if model_type != 'llama':
        raise ValueError("query capture currently supports Llama only")
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
