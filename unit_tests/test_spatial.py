"""Chunk rewrites and a RoPE round trip. No model load."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import spatial_pool
from engine.kv_compress.cache import patch_cache_update
from compression_topics.spatial.algorithms.spatial import apply_pool
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec


def _chunk() -> torch.Tensor:
    # [batch, heads, seq=10, dim=16]. Token 3 is the outlier. Tail of 2 stays exact.
    tokens = torch.zeros(1, 1, 10, 16)
    tokens[0, 0, :8] = 1
    tokens[0, 0, 3] = torch.arange(16, dtype=torch.float32) + 4
    tokens[0, 0, 8] = 7
    tokens[0, 0, 9] = 9
    return tokens


class SpatialRewriteTests(unittest.TestCase):
    def test_pool_replaces_closed_chunk_and_keeps_tail(self):
        source = _chunk()
        out = apply_pool(source, layer_idx=0, target="k", chunk=8)
        mean = source[0, 0, :8].mean(dim=0)
        self.assertTrue(torch.allclose(out[0, 0, :8], mean.expand(8, 16)))
        self.assertTrue(torch.equal(out[0, 0, 8:], source[0, 0, 8:]))

    def test_short_sequence_is_unchanged(self):
        source = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        out = apply_pool(source, layer_idx=0, target="k", chunk=8)
        self.assertTrue(torch.equal(out, source))

    def test_values_pool_along_the_feature_axis(self):
        source = torch.zeros(1, 1, 4, 10)
        source[0, 0, :, :8] = 1
        source[0, 0, :, 3] = torch.tensor([4.0, 5, 6, 7])
        source[0, 0, :, 8:] = 9
        out = apply_pool(source, layer_idx=0, target="v", chunk=8)
        mean = source[0, 0, :, :8].mean(dim=-1)
        self.assertTrue(torch.allclose(out[0, 0, :, :8], mean[:, None].expand(4, 8)))
        self.assertTrue(torch.equal(out[0, 0, :, 8:], source[0, 0, :, 8:]))


class RopeRoundTripTests(unittest.TestCase):
    def test_inverse_then_forward_restores_keys(self):
        rope = RopeTables(rope_theta=500000.0, head_dim=16)
        keys = torch.randn(1, 2, 6, 16)
        positions = torch.arange(6)
        cos, sin = rope.cos_sin(positions, keys.dtype)
        rotated = apply_rope(keys, cos, sin, inverse=False)
        restored = apply_rope(rotated, cos, sin, inverse=True)
        self.assertTrue(torch.allclose(restored, keys, atol=1e-5))


class SpatialCacheTests(unittest.TestCase):
    def test_cache_update_pools_keys_as_stored(self):
        from transformers.cache_utils import Cache, DynamicCache

        rope = RopeTables(rope_theta=10000.0, head_dim=4)
        spec = parse_kv_spec(spatial_pool(k_layers="all", v_layers=[], chunk=4))
        stored_keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        uninstall = patch_cache_update(spec, rope)
        try:
            cache = DynamicCache()
            stored, _values = Cache.update(cache, stored_keys.clone(), stored_keys.clone(), 0)
        finally:
            uninstall()
        expected = apply_pool(stored_keys, layer_idx=0, target="k", chunk=4)
        self.assertTrue(torch.allclose(stored, expected, atol=1e-5))


class SpatialRunTests(unittest.TestCase):
    def test_gsm8k_budgets_are_named_per_method(self):
        from catalog.tasks import GSM8K_20PCT
        from engine.layer_select.budgets import budget_run

        selection = {
            "tag": "spatial_pool_p15",
            "budget": 0.15,
            "compression": 0.15,
            "kv": spatial_pool(k_layers=[0], v_layers=[]),
        }
        names = [item["name"] for item in budget_run([selection], tasks=(GSM8K_20PCT,))["configurations"]]
        self.assertEqual(names, ["llama31_gsm8k_dense", "llama31_gsm8k_spatial_pool_p15"])


if __name__ == "__main__":
    unittest.main()
