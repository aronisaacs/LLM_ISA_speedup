"""The ``spatial`` KV compression method: spatial pair compression.

Adjacent tokens are merged in pairs per KV head. Every window of every head (32
tokens since 2026-11-21; a step without a ``window`` key uses 16, as all earlier
runs did) is compressed to the slot's saving, into a fixed-size block, as soon as its
last token is written to the cache, in the prompt and during generation alike
(``windows.py``). Tokens of an unfinished window stay dense until it completes.
Keys use the query-aware reconstruction error when query weights are given;
values always use the relative error.
"""

from dataclasses import asdict, dataclass, fields
import math

from compression_topics.spatial.algorithms import windows
from compression_topics.spatial.algorithms.formats import (
    ACCOUNTING,
    HEADER_BITS,
    TAG_BITS,
    OFFICIAL_WINDOW,
    WINDOW,
    dense_bits,
)
from compression_topics.spatial.algorithms.representation import SCALES, query_energy
from engine.kv_compress.methods import AFTER_APPEND, METHODS, NEEDS_PADDING, OPTION_VALIDATORS
from engine.kv_compress.padding import left_pads


@dataclass(frozen=True)
class SpatialConfig:
    query_weights: str | None = None  # calibrated E[q^2] file, for keys
    query_weights_sha256: str | None = None
    window: int = OFFICIAL_WINDOW  # tokens per window; written only when not 16, the identity of older runs
    scale: str = "length"  # stored per-token factor (representation.reconstruct); written only when not "length"

    def __post_init__(self):
        if (self.query_weights is None) != (self.query_weights_sha256 is None):
            raise ValueError("query weights need their checksum, and the checksum its weights")
        _check_window(self.window)
        if self.scale not in SCALES:
            raise ValueError(f"spatial scale must be one of {SCALES}")

    def pipeline(self, saving, *, k_layers="all", v_layers="all"):
        if not math.isfinite(saving) or not 0 < saving < 1:
            raise ValueError("saving must be between zero and one")
        options = {k: v for k, v in asdict(self).items()
                   if v is not None and not (k == "window" and v == WINDOW)
                   and not (k == "scale" and v == "length")}
        return {"pipeline": [dict(method="spatial", saving=saving, k_layers=k_layers,
                                  v_layers=v_layers, **options)]}


def _stats(target, layer_idx):
    from engine.kv_compress.metrics import STATS

    return STATS.setdefault(
        f"{target}_layer_{layer_idx}",
        {"accounting": ACCOUNTING, "dense_bits": 0, "stored_bits": 0, "metadata_bits": 0,
         "updates": 0},
    )


def store(tensor, *, saving, query_weights=None, query_weights_sha256=None, window=WINDOW, scale="length",
          **context):
    """New tokens are stored as written; :func:`after_append` compresses whole windows."""
    return tensor


# Tokens already compressed per (layer, target), per batch row, for the batch in progress. Counted
# from each row's first real token.
_DONE = {}


def after_append(cache_tensor, *, target, layer_idx, start, end, saving, rope_tables=None,
                 query_weights=None, query_weights_sha256=None, window=WINDOW, scale="length"):
    """Compress, in place, every ``window``-token window completed by this cache update.

    Each batch row's windows start at its first real token (left padding from the attention mask,
    ``engine/kv_compress/padding.py``), so a row is compressed as it would be in a batch of one.
    Padding is neither compressed nor counted.
    """
    if cache_tensor.ndim != 4:
        raise ValueError("spatial compression needs a [batch, KV heads, tokens, head_dim] cache")
    batch = cache_tensor.shape[0]
    pads = left_pads(batch, end)
    key = (layer_idx, target)
    if start == 0 or key not in _DONE or len(_DONE[key]) != batch:  # a new batch
        _DONE[key] = [0] * batch
    heads, dim = cache_tensor.shape[1], cache_tensor.shape[-1]
    row = _stats(target, layer_idx)
    for r, pad in enumerate(pads):
        appended = end - max(start, pad)  # real tokens of this row in this update
        if appended <= 0:
            continue
        new = heads * dense_bits(appended, dim)
        row["dense_bits"] += new
        row["stored_bits"] += new
        done = _DONE[key][r]
        complete = (end - pad) // window * window
        if complete <= done:
            continue
        body = cache_tensor[r:r + 1, :, pad + done:pad + complete, :]
        energy = None
        if target == "k" and query_weights is not None:
            energy = query_energy(query_weights, query_weights_sha256, layer_idx, body)
        out, bits, counts = windows.compress(body, target=target, saving=saving, query_energy=energy,
                                             window=window, scale=scale)
        cache_tensor[r:r + 1, :, pad + done:pad + complete, :] = out
        header = HEADER_BITS if done == 0 else 0
        row["stored_bits"] -= heads * dense_bits(complete - done, dim) - bits - header
        row["metadata_bits"] += header + TAG_BITS * sum(counts.values())
        row["updates"] += 1
        for name, count in counts.items():
            row[name] = row.get(name, 0) + count
        _DONE[key][r] = complete


def reset():
    """Forget per-sequence state (a new sequence also resets it automatically)."""
    _DONE.clear()


def _check_window(window):
    if isinstance(window, bool) or not isinstance(window, int) or window < 2 or window % 2:
        raise ValueError("spatial window must be an even number of tokens")


def _validate(options):
    names = {f.name for f in fields(SpatialConfig)}
    SpatialConfig(**{k: v for k, v in options.items() if k in names})


METHODS["spatial"] = store
AFTER_APPEND["spatial"] = after_append
OPTION_VALIDATORS["spatial"] = _validate
NEEDS_PADDING.add("spatial")
