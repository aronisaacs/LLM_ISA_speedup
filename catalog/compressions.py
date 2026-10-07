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


def sparsify_nm(k_layers="all", v_layers="all"):
    """Keep the 4 largest-magnitude values in each tile of 8. The ratio is fixed."""
    return _step("sparsify_nm", k_layers, v_layers)


def checksparse_l1(k_layers="all", v_layers="all", tile=8, prune_pct=50):
    """Zero the weakest L1 tiles (Mustafa's PyTorch checksparse stand-in)."""
    return _step("checksparse_l1", k_layers, v_layers, tile=tile, prune_pct=prune_pct)


def checksparse_row(k_layers="all", v_layers="all", tile=8, prune_pct=50):
    """Zero the weakest L1 tiles across all KV heads of each token (Mustafa's checksparse)."""
    return _step("checksparse_row", k_layers, v_layers, tile=tile, prune_pct=prune_pct)


def vector_compress(k_layers="all", v_layers="all", threshold=0.0, prune_pct=None):
    """Zero weak scalars. ``prune_pct`` drops that percent; otherwise ``|x| < threshold``."""
    if prune_pct is not None:
        return _step("vector_compress", k_layers, v_layers, prune_pct=int(prune_pct))
    return _step("vector_compress", k_layers, v_layers, threshold=threshold)


def vector_compress_pair(k_layers="all", v_layers="all", prune_pct=50):
    """Prune keys by RoPE-pair magnitude and values by per-scalar magnitude.

    A key pairs feature ``i`` with feature ``i + head_dim // 2``. Pair magnitude
    is ``a² + b²``. ``prune_pct`` zeros the weakest that percent of pairs.
    Values use the same per-scalar prune as ``vector_compress``.
    """
    return _step("vector_compress_pair", k_layers, v_layers, prune_pct=int(prune_pct))


def residual_pool(k_layers="all", v_layers="all", prune_pct=25, rope=False):
    """Pool adjacent tokens and keep a residual on the largest |delta| features.

    ``prune_pct`` zeros that percent of the residual. 100 is pure pooling.
    ``rope`` aligns each key pair by one RoPE step before the mean and delta.
    Values ignore it.
    """
    if not isinstance(rope, bool):
        raise ValueError("residual_pool rope must be a bool")
    return _step(
        "residual_pool",
        k_layers,
        v_layers,
        prune_pct=int(prune_pct),
        rope=rope,
    )


def pair_pool(k_layers="all", v_layers="all"):
    """Replace each adjacent token pair with its mean. No residual."""
    return _step("pair_pool", k_layers, v_layers)


def pair_quant(k_layers="all", v_layers="all", bits=8, group=32):
    """Pool adjacent tokens and keep the residual as ``bits``-bit integers.

    ``bits`` is 8, 4, 2, 1, or 0. 0 stores no residual, which is ``pair_pool``.
    One scale per ``group`` features. The sweep climber walks 8, 4, 2, 1, then
    merge-only, which is level 100 in a sweep because 0 already means dense.
    """
    return _step("pair_quant", k_layers, v_layers, bits=int(bits), group=int(group))


def pair_gate(k_layers="all", v_layers="all", tau=0.3, keep_pct=0):
    """Merge a token pair only when it is similar, keeping the biggest differences exactly.

    ``keep_pct`` percent of the features with the largest |difference| are stored exactly.
    The pair merges when the difference left over is at most ``tau`` times the pair's mean
    in length; otherwise both tokens stay as they are. ``keep_pct`` 0 keeps no residual.
    """
    return _step("pair_gate", k_layers, v_layers, tau=float(tau), keep_pct=int(keep_pct))


def spatial_pool(k_layers="all", v_layers="all", chunk=8):
    """Replace each closed chunk with its mean."""
    return _step("spatial_pool", k_layers, v_layers, chunk=chunk)


def quantize(k_layers="all", v_layers="all", bits=8, group=32):
    """Symmetric int8 or int4, one absmax scale per group of 32. Post-RoPE.

    Values group features inside a token. Keys group tokens inside a feature
    across the cache. A short tail stays full precision until the group fills.
    The sweep climber walks 8 bits, then 4.
    """
    return _step("quantize", k_layers, v_layers, bits=int(bits), group=int(group))


def qjl(k_layers="all", v_layers="all", bits=4):
    """Hadamard rotation and a fixed codebook, with an fp16 norm. Post-RoPE.

    The sweep climber overwrites ``bits`` per slot, walking 4, then 3, then 2, then 1.
    """
    return _step("qjl", k_layers, v_layers, bits=int(bits))


SPARSIFY_48 = sparsify_nm("all", "all")
CHECKSPARSE_L1_50 = checksparse_l1("all", "all", tile=8, prune_pct=50)
