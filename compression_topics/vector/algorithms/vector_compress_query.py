"""Query-aware vector compression: keys ranked by their effect on attention scores.

Plain ranking zeros the smallest k_d^2 (``vector_compress``) or a^2 + b^2 per
RoPE pair (``vector_compress_pair``). Here each score is weighted by the
calibrated query energy w of that layer and KV head:

* ``vector_compress_query``: zero the smallest w_d k_d^2;
* ``vector_compress_pair_query``: zero the pairs (i, i + d/2) with the smallest
  w_i (a^2 + b^2).

The weights are post-RoPE E[q^2] from the spatial study's calibration file,
loaded through its checksum check. That loader averages the two features of
each RoPE plane, so w_i = w_{i + d/2} and the weights do not depend on rotation.
Values are never rotated and never meet a query here: they use the plain
per-scalar prune, exactly as in the plain methods. Storage and the bitmask are
the same as the plain methods.
"""

from __future__ import annotations

import torch

from compression_topics.spatial.algorithms.representation import query_energy
from compression_topics.vector.algorithms.vector_compress import _apply as scalar_prune
from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS

# Tracked in git, so a pipeline naming it runs the same on the Mac and the DGX.
QUERY_WEIGHTS = "compression_topics/spatial/figures/clean_online_ladder_query_hierarchy/query_weights.pt"
QUERY_WEIGHTS_SHA256 = "d93ebef9e13556468178b09e25bf1ccba8ec354b49032878e22078b3ba0945d8"


def scalar_query(keys: torch.Tensor, weights: torch.Tensor, prune_pct: int) -> torch.Tensor:
    """Zero the prune_pct of values with the smallest w_d k_d^2. ``weights`` broadcasts over ``keys``."""
    n_drop = keys.shape[-1] * prune_pct // 100
    if n_drop == 0:
        return keys
    score = keys.float().square() * weights
    drop = score.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    return keys.scatter(-1, drop, 0)


def pair_query(keys: torch.Tensor, weights: torch.Tensor, prune_pct: int) -> torch.Tensor:
    """Zero the prune_pct of pairs (i, i + d/2) with the smallest (w_i + w_{i+d/2})(a^2 + b^2)."""
    half = keys.shape[-1] // 2
    n_drop = half * prune_pct // 100
    if n_drop == 0:
        return keys
    first, second = keys[..., :half], keys[..., half:]
    plane = weights[..., :half] + weights[..., half:]
    score = (first.float().square() + second.float().square()) * plane
    drop = score.topk(n_drop, dim=-1, largest=False, sorted=False).indices
    keep = torch.ones_like(score, dtype=torch.bool).scatter(-1, drop, False)
    return torch.cat((first * keep, second * keep), dim=-1)


def _weights(layer_idx, keys, query_weights, query_weights_sha256):
    energy = query_energy(query_weights, query_weights_sha256, layer_idx, keys)
    return energy.reshape(1, keys.shape[1], 1, keys.shape[-1])


def _method(rank):
    def apply(tensor, *, layer_idx, target, prune_pct=0, query_weights=QUERY_WEIGHTS,
              query_weights_sha256=QUERY_WEIGHTS_SHA256, **_unused):
        if target != "k":
            return scalar_prune(tensor, threshold=0.0, prune_pct=prune_pct)
        return rank(tensor, _weights(layer_idx, tensor, query_weights, query_weights_sha256), prune_pct)
    return apply


def _validate(options):
    prune_pct = options.get("prune_pct")
    if not isinstance(prune_pct, int) or isinstance(prune_pct, bool) or not 0 <= prune_pct <= 100:
        raise ValueError("query-aware vector compression needs an integer prune_pct 0..100")
    if not options.get("query_weights") or not options.get("query_weights_sha256"):
        raise ValueError("query-aware vector compression needs query_weights and query_weights_sha256")


METHODS["vector_compress_query"] = _method(scalar_query)
METHODS["vector_compress_pair_query"] = _method(pair_query)
OPTION_VALIDATORS["vector_compress_query"] = _validate
OPTION_VALIDATORS["vector_compress_pair_query"] = _validate
