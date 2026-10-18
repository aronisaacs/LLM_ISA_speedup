"""Capture post-RoPE queries next to Cache.update (used to calibrate query energy)."""


MODULES = {
    'llama': 'transformers.models.llama.modeling_llama',
    'qwen3_5': 'transformers.models.qwen3_5.modeling_qwen3_5',
}


def capture_queries(model_type):
    """A one-layer handoff from the model's rotary to Cache.update; restored on uninstall."""
    if model_type not in MODULES:
        raise ValueError(f"query capture supports {', '.join(MODULES)} only")
    import importlib
    llama = importlib.import_module(MODULES[model_type])
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
