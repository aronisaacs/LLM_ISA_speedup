"""kv.pipeline dicts used by run files.

Each helper returns {"pipeline": [one step]}. Dense is an empty pipeline.
Layer lists stay here so a run can say sparsify_nm([20], []) without copying JSON.
"""

from __future__ import annotations

from typing import Any


DENSE: dict[str, Any] = {"pipeline": []}


def _step(method: str, k_layers, v_layers, **kwargs) -> dict[str, Any]:
    payload = {"method": method, "k_layers": k_layers, "v_layers": v_layers}
    payload.update(kwargs)
    return {"pipeline": [payload]}


def sparsify_nm(k_layers="all", v_layers="all", n=8, m=4):
    """Keep m largest-magnitude values in each tile of n (default 4:8)."""
    return _step("sparsify_nm", k_layers, v_layers, n=n, m=m)


def checksparse_l1(k_layers="all", v_layers="all", tile=8, prune_pct=50):
    """Zero the weakest L1 tiles (Mustafa's PyTorch checksparse stand-in)."""
    return _step("checksparse_l1", k_layers, v_layers, tile=tile, prune_pct=prune_pct)


def vector_compress(k_layers="all", v_layers="all", threshold=0.0, prune_pct=None):
    """Zero weak scalars. ``prune_pct`` drops that percent; otherwise ``|x| < threshold``."""
    if prune_pct is not None:
        return _step("vector_compress", k_layers, v_layers, prune_pct=int(prune_pct))
    return _step("vector_compress", k_layers, v_layers, threshold=threshold)


def dynamic_precision(k_layers="all", v_layers="all", tile=8, bits=(16, 8, 4), pcts=(25, 50, 25)):
    """Rank tiles by L1; loudest `pcts[i]` percent get `bits[i]` fake-quantized."""
    return _step(
        "dynamic_precision",
        k_layers,
        v_layers,
        tile=tile,
        bits=list(bits),
        pcts=list(pcts),
    )


def spatial_pool(k_layers="all", v_layers="all", chunk=8, pre_rope=True):
    """Replace each closed chunk with its mean. Keys undo RoPE unless ``pre_rope`` is false."""
    return _spatial("spatial_pool", k_layers, v_layers, chunk, pre_rope)


def spatial_top1(k_layers="all", v_layers="all", chunk=8, pre_rope=True):
    """Keep the token farthest from the chunk mean; the rest become the mean."""
    return _spatial("spatial_top1", k_layers, v_layers, chunk, pre_rope)


def spatial_tile(k_layers="all", v_layers="all", chunk=8, tile=8, pre_rope=True):
    """Per tile, keep the token whose tile is farthest from the mean tile."""
    return _spatial("spatial_tile", k_layers, v_layers, chunk, pre_rope, tile=tile)


def spatial_feature(k_layers="all", v_layers="all", chunk=8, pre_rope=True):
    """Per feature, keep the token with the largest absolute deviation from the mean."""
    return _spatial("spatial_feature", k_layers, v_layers, chunk, pre_rope)


def qjl(k_layers="all", v_layers="all", bits=4):
    """Hadamard rotation and a fixed codebook, with an fp16 norm. Post-RoPE."""
    return _step("qjl", k_layers, v_layers, bits=int(bits))


def _spatial(method, k_layers, v_layers, chunk, pre_rope, **kwargs):
    payload = _step(method, k_layers, v_layers, chunk=chunk, **kwargs)
    if pre_rope:
        payload["pipeline"][0]["pre_rope"] = True
    return payload


SPARSIFY_48 = sparsify_nm("all", "all", n=8, m=4)
CHECKSPARSE_L1_50 = checksparse_l1("all", "all", tile=8, prune_pct=50)
DYNAMIC_PRECISION_1684 = dynamic_precision("all", "all")
