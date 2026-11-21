"""Left padding of the batch in progress, for methods that compress groups of tokens.

A method that groups consecutive tokens (spatial pairs and windows) must align its groups at each
row's first real token, as in a batch of one. ``install`` records the attention mask of every forward
call of the model for such methods (``NEEDS_PADDING``); the cache hook asks :func:`left_pads`.

lm-eval passes a mask to ``generate`` (left padding) but calls the model without one for
log-likelihoods (right padding): there a batch above one cannot be aligned and is refused.
"""

from __future__ import annotations

_mask = None  # attention mask [batch, tokens] of the forward call in progress, or None


def record(module, args, kwargs) -> None:
    """Forward pre-hook (``with_kwargs=True``) on the model."""
    global _mask
    del module, args
    _mask = kwargs.get("attention_mask")


def clear() -> None:
    global _mask
    _mask = None


def left_pads(batch: int, end: int) -> list[int]:
    """Leading padding tokens of each row, for a cache holding ``end`` positions.

    No mask: a batch of one has none; a larger batch is refused (its padding is unknown).
    """
    mask = _mask
    if mask is None or getattr(mask, "ndim", 0) != 2:
        if batch == 1:
            return [0]
        raise ValueError("a batch above one needs a 2-D attention mask to find each row's padding "
                         "(lm-eval log-likelihood calls pass none: run them with batch size 1)")
    if tuple(mask.shape) != (batch, end):
        raise ValueError(f"attention mask {tuple(mask.shape)} does not cover the cache ({batch}, {end})")
    real = mask.bool()
    pads = (real.long().cumsum(-1) == 0).sum(-1)
    width = real.sum(-1)
    if not bool((pads + width == end).all()):
        raise ValueError("only left padding is supported (a row has padding after real tokens)")
    return [int(p) for p in pads.tolist()]
