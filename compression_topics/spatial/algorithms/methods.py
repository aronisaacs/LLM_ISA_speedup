"""Spatial compression methods: ``spatial_offline``, ``spatial_online`` and ``spatial_chunked``.

Both merge adjacent tokens per KV head (pairs, and optionally quads/eights) with
the same preprocessing, formats and loss; they differ only in how the price is set.

* offline: one price per layer slot, fitted over the whole prompt so the slot meets
  its saving exactly. It needs the full prompt, so tokens generated afterwards
  stay dense.
* online: the price is fitted on the first 32 tokens and then steered after every
  eight-token window. Each window is compressed as soon as its eighth token is
  written, in the prompt and during generation alike; the controller state is
  kept per layer slot across cache updates. Tokens of an unfinished window (and
  all tokens before the first 32 arrive) stay dense until the window completes.

* chunked: no price is carried between windows. Each completed eight-token
  window, in the prompt and during generation, is compressed on its own to the
  slot's saving (``chunked.py``). Tokens of an unfinished window stay dense.

Query awareness applies to keys only; values always use the relative loss.
"""

from dataclasses import asdict, dataclass, fields
import math

from compression_topics.spatial.algorithms import chunked, windows
from compression_topics.spatial.algorithms.controller import (
    Control,
    Online,
    SATURATED_PRICE,
    offline_price,
)
from compression_topics.spatial.algorithms.formats import (
    ACCOUNTING,
    HEADER_BITS,
    MENUS,
    WINDOW,
    dense_bits,
    validate_layout,
)
from compression_topics.spatial.algorithms.representation import query_energy
from engine.kv_compress.methods import AFTER_APPEND, METHODS, OPTION_VALIDATORS

ALGORITHMS = ("offline", "online", "chunked")


@dataclass(frozen=True)
class SpatialConfig:
    algorithm: str = "online"
    max_group: int = 8
    query_aware: bool = False
    query_weights: str | None = None
    query_weights_sha256: str | None = None
    control: dict | None = None
    fixed_residual: int | None = None
    minimum_storage: bool = False
    residual_counts: dict | None = None
    prefill_only: bool = False
    merge_sizes: tuple | None = None

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError("algorithm must be offline, online or chunked")
        validate_layout(
            self.max_group,
            fixed_residual=self.fixed_residual,
            residual_counts=self.residual_counts,
            merge_sizes=self.merge_sizes,
        )
        if not isinstance(self.query_aware, bool) or not isinstance(
            self.minimum_storage, bool
        ):
            raise ValueError("query_aware and minimum_storage are booleans")
        if self.query_aware != (
            self.query_weights is not None and self.query_weights_sha256 is not None
        ):
            raise ValueError(
                "query-aware loss requires calibrated weights and their checksum, and only then"
            )
        if self.minimum_storage and (self.algorithm == "offline" or self.max_group != 2):
            raise ValueError("the minimum-storage endpoint is online or chunked pairs only")
        if not isinstance(self.prefill_only, bool):
            raise ValueError("prefill_only must be boolean")
        if self.control is not None:
            if self.algorithm != "online":
                raise ValueError("only the online algorithm uses controller settings")
            Control(**self.control)

    def pipeline(self, saving, *, k_layers="all", v_layers="all"):
        if not math.isfinite(saving) or saving <= 0:
            raise ValueError("saving must be positive")
        options = asdict(self)
        if options["merge_sizes"] is not None:
            options["merge_sizes"] = list(options["merge_sizes"])
        algorithm = options.pop("algorithm")
        return {
            "pipeline": [
                {
                    "method": "spatial_" + algorithm,
                    "saving": saving,
                    "k_layers": k_layers,
                    "v_layers": v_layers,
                    **options,
                }
            ]
        }


def _energy(
    target, layer_idx, tensor, query_aware, query_weights, query_weights_sha256
):
    if not query_aware or target != "k":
        return None
    return query_energy(query_weights, query_weights_sha256, layer_idx, tensor)


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
            "controller_bound_hits": 0,
            "max_final_target_deviation": 0.0,
        },
    )


def _record_formats(row, choices):
    for choice in choices:
        for head in choice.heads:
            for _, size, residual in head.segments:
                name = (
                    "format_dense_pair"
                    if residual is None
                    else f"format_g{size}_r{residual}"
                )
                row[name] = row.get(name, 0) + 1


def offline(
    tensor,
    *,
    layer_idx,
    target,
    saving,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    control=None,
    fixed_residual=None,
    minimum_storage=False,
    seq_start=0,
    rope_tables=None,
    residual_counts=None,
    prefill_only=False,
    merge_sizes=None,
):
    """Whole-prompt price; generated tokens (seq_start > 0) are left dense."""
    SpatialConfig(
        "offline",
        max_group,
        query_aware,
        query_weights,
        query_weights_sha256,
        control,
        fixed_residual,
        minimum_storage,
        residual_counts,
        prefill_only,
        merge_sizes,
    )
    heads, length, dim = tensor.shape[1:]
    row = _stats(target, layer_idx)
    row["dense_bits"] += heads * dense_bits(length, dim)
    row["stored_bits"] += heads * dense_bits(length, dim)
    full = length // WINDOW * WINDOW
    if seq_start or not full:
        return tensor
    body = tensor[..., :full, :]
    energy = _energy(
        target, layer_idx, body, query_aware, query_weights, query_weights_sha256
    )
    table, recon = windows.build(
        body,
        target=target,
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=energy,
        fixed_residual=fixed_residual,
        residual_counts=residual_counts,
        merge_sizes=merge_sizes,
    )
    price = offline_price(
        table, saving, tail_bits=heads * dense_bits(length - full, dim)
    )
    choices = [table.choose(w, price) for w in range(len(table))]
    out = tensor.clone()
    out[..., :full, :] = windows.restore(body, choices, recon)
    stored = sum(c.bits for c in choices) + HEADER_BITS
    row["stored_bits"] -= len(table) * table.window_bits - stored
    row["metadata_bits"] += HEADER_BITS + sum(c.tags for c in choices) * 3
    row["updates"] += 1
    measured = 1 - (stored + heads * dense_bits(length - full, dim)) / (
        heads * dense_bits(length, dim)
    )
    row["max_final_target_deviation"] = max(
        row["max_final_target_deviation"], abs(measured - saving)
    )
    _record_formats(row, choices)
    return out


# Online state per layer slot for the sequence being processed (batch size one).
_STATE = {}


class _Stream:
    def __init__(self, saving, control, minimum_storage):
        self.controller = Online(saving, Control(**(control or {})), minimum_storage)
        self.done = 0  # tokens already decided (a multiple of eight)
        self.dense_tokens = 0  # tokens seen in this sequence


def online(
    tensor,
    *,
    layer_idx,
    target,
    saving,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    control=None,
    fixed_residual=None,
    minimum_storage=False,
    seq_start=0,
    rope_tables=None,
    residual_counts=None,
    prefill_only=False,
    merge_sizes=None,
):
    """New tokens are stored as written; ``online_after_append`` compresses completed windows."""
    return tensor


def online_after_append(
    cache_tensor,
    *,
    target,
    layer_idx,
    start,
    end,
    rope_tables=None,
    saving,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    control=None,
    fixed_residual=None,
    minimum_storage=False,
    residual_counts=None,
    prefill_only=False,
    merge_sizes=None,
):
    """Compress, in place in the cache, every eight-token window completed by this update."""
    if cache_tensor.ndim != 4 or cache_tensor.shape[0] != 1:
        raise ValueError("spatial compression requires batch size one")
    key = (layer_idx, target)
    if start == 0 or key not in _STATE:
        _STATE[key] = _Stream(saving, control, minimum_storage)
    stream = _STATE[key]
    heads, dim = cache_tensor.shape[1], cache_tensor.shape[-1]
    row = _stats(target, layer_idx)
    new = heads * dense_bits(end - start, dim)
    row["dense_bits"] += new
    row["stored_bits"] += new
    stream.dense_tokens = end
    if prefill_only and start > 0:
        return  # Preserve the historical prefill-only evaluation protocol.
    complete = end // WINDOW * WINDOW
    controller = stream.controller
    if not controller.started and complete < controller.warmup * WINDOW:
        return  # wait for the startup tokens; they stay dense meanwhile
    if complete <= stream.done:
        return
    body = cache_tensor[..., stream.done : complete, :]
    energy = _energy(
        target, layer_idx, body, query_aware, query_weights, query_weights_sha256
    )
    table, recon = windows.build(
        body,
        target=target,
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=energy,
        fixed_residual=fixed_residual,
        residual_counts=residual_counts,
        merge_sizes=merge_sizes,
    )
    first = not controller.started
    if first:
        controller.start(table)
    hits = controller.hits
    choices = [controller.decide(table, w)[0] for w in range(len(table))]
    cache_tensor[..., stream.done : complete, :] = windows.restore(body, choices, recon)
    bits = sum(c.bits for c in choices) + (HEADER_BITS if first else 0)
    row["stored_bits"] -= len(table) * table.window_bits - bits
    row["metadata_bits"] += (HEADER_BITS if first else 0) + 3 * sum(
        c.tags for c in choices
    )
    row["updates"] += 1
    row["controller_bound_hits"] += controller.hits - hits
    tail = heads * dense_bits(end - complete, dim)
    measured = 1 - (controller.stored + tail) / (controller.dense + tail)
    row["max_final_target_deviation"] = max(
        row["max_final_target_deviation"], abs(measured - saving)
    )
    _record_formats(row, choices)
    stream.done = complete


# Tokens already compressed per layer slot by the chunked method.
_CHUNKED = {}


def chunked_after_append(
    cache_tensor,
    *,
    target,
    layer_idx,
    start,
    end,
    rope_tables=None,
    saving,
    max_group=8,
    query_aware=False,
    query_weights=None,
    query_weights_sha256=None,
    control=None,
    fixed_residual=None,
    minimum_storage=False,
    residual_counts=None,
    prefill_only=False,
    merge_sizes=None,
):
    """Compress, in place, every eight-token window completed by this update to ``saving``."""
    if cache_tensor.ndim != 4 or cache_tensor.shape[0] != 1:
        raise ValueError("spatial compression requires batch size one")
    key = (layer_idx, target)
    first = start == 0 or key not in _CHUNKED
    if first:
        _CHUNKED[key] = 0
    done = _CHUNKED[key]
    heads, dim = cache_tensor.shape[1], cache_tensor.shape[-1]
    row = _stats(target, layer_idx)
    new = heads * dense_bits(end - start, dim)
    row["dense_bits"] += new
    row["stored_bits"] += new
    if prefill_only and start > 0:
        return
    complete = end // WINDOW * WINDOW
    if complete <= done:
        return
    body = cache_tensor[..., done:complete, :]
    energy = _energy(
        target, layer_idx, body, query_aware, query_weights, query_weights_sha256
    )
    out, bits, counts = chunked.compress(
        body,
        target=target,
        saving=saving,
        max_group=max_group,
        rope_tables=rope_tables,
        query_energy=energy,
        fixed_residual=fixed_residual,
        residual_counts=residual_counts,
        merge_sizes=merge_sizes,
        minimum_storage=minimum_storage,
    )
    cache_tensor[..., done:complete, :] = out
    header = HEADER_BITS if done == 0 else 0
    row["stored_bits"] -= heads * dense_bits(complete - done, dim) - bits - header
    row["metadata_bits"] += header + 3 * sum(counts.values())
    row["updates"] += 1
    for name, count in counts.items():
        row[name] = row.get(name, 0) + count
    _CHUNKED[key] = complete


def reset():
    """Forget online state (a new sequence also resets it automatically)."""
    _STATE.clear()
    _CHUNKED.clear()


def _validate(options):
    names = {f.name for f in fields(SpatialConfig)}
    SpatialConfig(**{k: v for k, v in options.items() if k in names})


METHODS["spatial_offline"] = offline
METHODS["spatial_online"] = online
AFTER_APPEND["spatial_online"] = online_after_append
METHODS["spatial_chunked"] = online  # stored as written; the append hook compresses
AFTER_APPEND["spatial_chunked"] = chunked_after_append
OPTION_VALIDATORS["spatial_offline"] = lambda options: _validate(
    dict(options, algorithm="offline")
)
OPTION_VALIDATORS["spatial_online"] = lambda options: _validate(
    dict(options, algorithm="online")
)
OPTION_VALIDATORS["spatial_chunked"] = lambda options: _validate(
    dict(options, algorithm="chunked")
)
