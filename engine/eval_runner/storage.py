"""Explicit storage scopes for token-loss evaluation and historical calibration.

New results carry prefix slots directly. The legacy reader below is the sole
place that converts old streaming counters that included dense continuation.
"""

import math


def byte_counts(row):
    dense, stored = row["dense_bits"], row["stored_bits"]
    if not all(
        isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
        for x in (dense, stored)
    ):
        raise ValueError("storage bits must be finite nonnegative numbers")
    return {
        "dense_bits": dense,
        "stored_bits": stored,
        "compression": 1 - stored / dense if dense else 0.0,
    }


def sum_storage(summaries):
    """Combine matching scopes; recompute rates rather than averaging them."""
    if not summaries:
        return None
    scope = summaries[0]["scope"]
    if any(s["scope"] != scope for s in summaries):
        raise ValueError("storage scopes differ")
    targets = {
        target: byte_counts(
            {
                field: sum(s["targets"][target][field] for s in summaries)
                for field in ("dense_bits", "stored_bits")
            }
        )
        for target in ("k", "v")
    }
    targets["kv"] = byte_counts(
        {
            field: sum(targets[t][field] for t in ("k", "v"))
            for field in ("dense_bits", "stored_bits")
        }
    )
    result = {"scope": scope, "targets": targets}
    if all("slots" in s for s in summaries):
        keys = set().union(*(s["slots"] for s in summaries))
        result["slots"] = {
            key: byte_counts(
                {
                    field: sum(s["slots"].get(key, {}).get(field, 0) for s in summaries)
                    for field in ("dense_bits", "stored_bits")
                }
            )
            for key in sorted(keys)
        }
    return result


def storage_delta(before, after):
    """Per-forward prefix storage from cumulative process counters."""
    targets = {
        target: byte_counts(
            {
                field: after["targets"][target][field]
                - (before["targets"][target][field] if before else 0)
                for field in ("dense_bits", "stored_bits")
            }
        )
        for target in ("k", "v")
    }
    result = {"scope": "simulated_prefill_cache_storage", "targets": targets}
    if "slots" in after:
        result["slots"] = {
            key: byte_counts(
                {
                    field: row[field]
                    - (
                        before.get("slots", {}).get(key, {}).get(field, 0)
                        if before
                        else 0
                    )
                    for field in ("dense_bits", "stored_bits")
                }
            )
            for key, row in after["slots"].items()
        }
    return result


def calibration_slot_bytes(payload, key):
    """Prefer explicit prefix slots; preserve old on-disk experiment identities."""
    storage = payload.get("storage", {})
    if "slots" in storage:
        if (
            payload.get("scoring", {}).get("prefix_tokens") is not None
            and storage.get("scope") != "simulated_prefill_cache_storage"
        ):
            raise ValueError(
                "continuation calibration requires explicit prefill storage"
            )
        if key not in storage["slots"]:
            raise ValueError("selected calibration slot missing from storage")
        return byte_counts(storage["slots"][key]), "explicit_storage_slots"
    row = payload["gate_stats"][key]
    dense, stored = row["dense_bits"], row["stored_bits"]
    steps = payload.get("simulation", {}).get("kv", {}).get("pipeline", [])
    prefix = payload.get("scoring", {}).get("prefix_tokens")
    if prefix is not None and any(
        s.get("method") == "spatial_online" and s.get("prefill_only") for s in steps
    ):
        scores = payload.get("chunk_scores", [])
        if (
            not scores
            or prefix <= 0
            or storage.get("scope") != "simulated_prefill_cache_storage"
        ):
            raise ValueError("missing historical prefill accounting")
        prefix_bits = (
            dense
            * len(scores)
            * prefix
            / (len(scores) * prefix + sum(r["tokens"] for r in scores))
        )
        return (
            byte_counts(
                {
                    "dense_bits": prefix_bits,
                    "stored_bits": prefix_bits - (dense - stored),
                }
            ),
            "historical_streaming_prefill_conversion",
        )
    return byte_counts(row), "historical_algorithm_counters"


def dense_storage(config, tokens, scope):
    """Dense baseline accounting uses the same slot schema as compressed runs."""
    config = getattr(config, "text_config", config)
    dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    bits = tokens * heads * dim * 16
    slots = {
        f"{target}_layer_{layer}": byte_counts(
            {"dense_bits": bits, "stored_bits": bits}
        )
        for target in ("k", "v")
        for layer in range(config.num_hidden_layers)
    }
    target_bits = bits * config.num_hidden_layers
    return sum_storage(
        [
            {
                "scope": scope,
                "slots": slots,
                "targets": {
                    target: {"dense_bits": target_bits, "stored_bits": target_bits}
                    for target in ("k", "v")
                },
            }
        ]
    )
