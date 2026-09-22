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


def vector_compress(k_layers="all", v_layers="all", threshold=0.0):
    """Zero each scalar whose magnitude is below threshold."""
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


SPARSIFY_48 = sparsify_nm("all", "all", n=8, m=4)
CHECKSPARSE_L1_50 = checksparse_l1("all", "all", tile=8, prune_pct=50)
DYNAMIC_PRECISION_1684 = dynamic_precision("all", "all")
