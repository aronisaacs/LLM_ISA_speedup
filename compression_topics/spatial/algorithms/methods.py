"""The ``spatial`` KV compression method: spatial pair compression.

Adjacent tokens are merged in pairs per KV head. Every 16-token window of every
head is compressed to the slot's saving, into a fixed-size block, as soon as its
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
    WINDOW,
    dense_bits,
)
from compression_topics.spatial.algorithms.representation import query_energy
from engine.kv_compress.methods import AFTER_APPEND, METHODS, OPTION_VALIDATORS


@dataclass(frozen=True)
class SpatialConfig:
    query_weights: str | None = None  # calibrated E[q^2] file, for keys
    query_weights_sha256: str | None = None

    def __post_init__(self):
        if (self.query_weights is None) != (self.query_weights_sha256 is None):
            raise ValueError("query weights need their checksum, and the checksum its weights")

    def pipeline(self, saving, *, k_layers="all", v_layers="all"):
        if not math.isfinite(saving) or not 0 < saving < 1:
            raise ValueError("saving must be between zero and one")
        options = {k: v for k, v in asdict(self).items() if v is not None}
        return {"pipeline": [dict(method="spatial", saving=saving, k_layers=k_layers,
                                  v_layers=v_layers, **options)]}


def _stats(target, layer_idx):
    from engine.kv_compress.metrics import STATS

    return STATS.setdefault(
        f"{target}_layer_{layer_idx}",
        {"accounting": ACCOUNTING, "dense_bits": 0, "stored_bits": 0, "metadata_bits": 0,
         "updates": 0},
    )


def store(tensor, *, saving, query_weights=None, query_weights_sha256=None, **context):
    """New tokens are stored as written; :func:`after_append` compresses whole windows."""
    return tensor


# Tokens already compressed per (layer, target) for the sequence in progress.
_DONE = {}


def after_append(cache_tensor, *, target, layer_idx, start, end, saving, rope_tables=None,
                 query_weights=None, query_weights_sha256=None):
    """Compress, in place, every 16-token window completed by this cache update."""
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
    complete = end // WINDOW * WINDOW
    if complete <= done:
        return
    body = cache_tensor[..., done:complete, :]
    energy = None
    if target == "k" and query_weights is not None:
        energy = query_energy(query_weights, query_weights_sha256, layer_idx, body)
    out, bits, counts = windows.compress(body, target=target, saving=saving, query_energy=energy)
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
