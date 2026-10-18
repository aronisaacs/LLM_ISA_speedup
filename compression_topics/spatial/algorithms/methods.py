"""The ``spatial`` KV compression method.

Adjacent tokens are merged per KV head (pairs, and optionally quads and eights).
Every eight-token window is compressed to the slot's saving as soon as its last
token is written to the cache, in the prompt and during generation alike
(``windows.py``). Tokens of an unfinished window stay dense until it completes.
Query awareness applies to keys only; values always use the relative loss.
"""

from dataclasses import asdict, dataclass, fields
import math

from compression_topics.spatial.algorithms import windows
from compression_topics.spatial.algorithms.formats import (
    ACCOUNTING,
    HEADER_BITS,
    TAG_BITS,
    WINDOW,
    dense_bits,
    sizes,
)
from compression_topics.spatial.algorithms.representation import query_energy
from engine.kv_compress.methods import AFTER_APPEND, METHODS, OPTION_VALIDATORS


@dataclass(frozen=True)
class SpatialConfig:
    max_group: int = 8
    query_aware: bool = False
    query_weights: str | None = None
    query_weights_sha256: str | None = None
    # Simplification studies; omitted from the pipeline while at their defaults.
    per_head_budget: bool = False  # each head meets the saving alone (pairs only)
    align_rope: bool = True  # undo RoPE on keys before merging
    keep_norms: bool = True  # store per-token norms and merge unit directions
    window: int = WINDOW  # tokens decided together; 16 is supported for pairs
    head_savings: tuple | None = None  # fixed saving per KV head (pairs only)
    pad_heads: bool = False  # count each head's reserved budget, not the bits it used

    def __post_init__(self):
        sizes(self.max_group)
        for flag in ("query_aware", "per_head_budget", "align_rope", "keep_norms"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} is a boolean")
        if self.per_head_budget and self.max_group != 2:
            raise ValueError("per-head budgets are implemented for pairs only")
        if self.window not in (WINDOW, 2 * WINDOW) or isinstance(self.window, bool):
            raise ValueError("window must be 8 or 16 tokens")
        if self.window != WINDOW and self.max_group != 2:
            raise ValueError("16-token windows are implemented for pairs only")
        if self.head_savings is not None:
            values = list(self.head_savings)
            if self.max_group != 2 or not values or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1
                for v in values
            ):
                raise ValueError("head_savings is one saving in [0, 1) per KV head, pairs only")
        if not isinstance(self.pad_heads, bool) or (
            self.pad_heads and self.head_savings is None and not self.per_head_budget
        ):
            raise ValueError("pad_heads is a boolean and needs per-head budgets")
        if self.query_aware != (
            self.query_weights is not None and self.query_weights_sha256 is not None
        ):
            raise ValueError(
                "query-aware loss requires calibrated weights and their checksum, and only then"
            )

    def pipeline(self, saving, *, k_layers="all", v_layers="all"):
        if not math.isfinite(saving) or not 0 < saving < 1:
            raise ValueError("saving must be between zero and one")
        step = dict(method="spatial", saving=saving, k_layers=k_layers, v_layers=v_layers)
        options = {k: v for k, v in asdict(self).items()
                   if k not in _STUDY_DEFAULTS or v != _STUDY_DEFAULTS[k]}
        if options.get("head_savings") is not None:
            options["head_savings"] = [float(v) for v in options["head_savings"]]
        return {"pipeline": [dict(step, **options)]}


_STUDY_DEFAULTS = dict(
    per_head_budget=False, align_rope=True, keep_norms=True, window=WINDOW, head_savings=None,
    pad_heads=False,
)


def _stats(target, layer_idx):
    from engine.kv_compress.metrics import STATS

    return STATS.setdefault(
        f"{target}_layer_{layer_idx}",
        {
            "accounting": ACCOUNTING,
            "dense_bits": 0,
            "stored_bits": 0,
            "metadata_bits": 0,
            "updates": 0,
        },
    )


def store(
    tensor,
    *,
    saving,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    per_head_budget=False,
    align_rope=True,
    keep_norms=True,
    window=WINDOW,
    head_savings=None,
    pad_heads=False,
    **context,
):
    """New tokens are stored as written; :func:`after_append` compresses whole windows."""
    return tensor


# Tokens already compressed per (layer, target) for the sequence in progress.
_DONE = {}


def after_append(
    cache_tensor,
    *,
    target,
    layer_idx,
    start,
    end,
    saving,
    rope_tables=None,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    per_head_budget=False,
    align_rope=True,
    keep_norms=True,
    window=WINDOW,
    head_savings=None,
    pad_heads=False,
):
    """Compress, in place, every window (8 or 16 tokens) completed by this cache update."""
    if cache_tensor.ndim != 4 or cache_tensor.shape[0] != 1:
        raise ValueError("spatial compression requires batch size one")
    key = (layer_idx, target)
    if start == 0 or key not in _DONE:  # a new sequence
        _DONE[key] = 0
    done = _DONE[key]
    heads, dim = cache_tensor.shape[1], cache_tensor.shape[-1]
    row = _stats(target, layer_idx)
    new = heads * dense_bits(end - start, dim)
    row["dense_bits"] += new
    row["stored_bits"] += new
    complete = end // window * window
    if complete <= done:
        return
    body = cache_tensor[..., done:complete, :]
    energy = None
    if query_aware and target == "k":
        energy = query_energy(query_weights, query_weights_sha256, layer_idx, body)
    out, bits, counts = windows.compress(
        body,
        target=target,
        saving=saving if head_savings is None else list(head_savings),
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=energy,
        align_rope=align_rope,
        keep_norms=keep_norms,
        per_head_budget=per_head_budget or head_savings is not None,
        window=window,
        pad_heads=pad_heads,
    )
    cache_tensor[..., done:complete, :] = out
    header = HEADER_BITS if done == 0 else 0
    row["stored_bits"] -= heads * dense_bits(complete - done, dim) - bits - header
    row["metadata_bits"] += header + TAG_BITS * sum(counts.values())
    row["updates"] += 1
    for name, count in counts.items():
        row[name] = row.get(name, 0) + count
    _DONE[key] = complete


def reset():
    """Forget per-sequence state (a new sequence also resets it automatically)."""
    _DONE.clear()


def _validate(options):
    names = {f.name for f in fields(SpatialConfig)}
    SpatialConfig(**{k: v for k, v in options.items() if k in names})


METHODS["spatial"] = store
AFTER_APPEND["spatial"] = after_append
OPTION_VALIDATORS["spatial"] = _validate
