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


def quantize(k_layers="all", v_layers="all", bits=8, group=32):
    """Symmetric int8 or int4, one absmax scale per group of 32. Post-RoPE.

    Values group features inside a token. Keys group tokens inside a feature
    across the cache. A short tail stays full precision until the group fills.
    The sweep climber walks 8 bits, then 4.
    """
    return _step("quantize", k_layers, v_layers, bits=int(bits), group=int(group))


# Calibrated outlier channels for Llama 3.1 8B (compression_topics/turboquant_mse/scripts/calibrate_outliers.py).
TURBOQUANT_OUTLIER_TABLE = "compression_topics/turboquant_mse/calibration/llama31_8b_outliers.json"


def turboquant_mse(k_layers="all", v_layers="all", bits=4, outliers=0, outlier_bits=None, outlier_table=None,
                   signs="rope_pairs"):
    """Randomized Hadamard rotation and a fixed codebook, with an fp16 norm. Post-RoPE.

    The sweep climber sets the level per slot: dense, then 4, 3, 2.5 (2 bits + 64 outlier
    channels at 3) and 2 bits. ``outliers`` > 0 quantizes that many calibrated channels per
    head separately at ``outlier_bits`` (default ``bits`` + 1). ``signs="rope_pairs"`` shares
    one random sign per RoPE pair; ``signs="independent"`` writes no ``signs`` key, the
    identity of runs made before that option existed. The split options are only written when used.
    """
    options = {} if signs == "independent" else {"signs": str(signs)}
    if outliers:
        options.update(outliers=int(outliers), outlier_table=str(outlier_table))
        if outlier_bits is not None:
            options["outlier_bits"] = int(outlier_bits)
    return _step("turboquant_mse", k_layers, v_layers, bits=int(bits), **options)


SPARSIFY_48 = sparsify_nm("all", "all")
CHECKSPARSE_L1_50 = checksparse_l1("all", "all", tile=8, prune_pct=50)
