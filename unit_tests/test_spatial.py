"""Chunk rewrites, inverse RoPE, and the spatial WikiText sweep. No model load."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

import json
import tempfile

from catalog.compressions import spatial_pool
from engine.kv_compress.cache import patch_cache_update
from engine.kv_compress.methods.spatial import apply_feature, apply_pool, apply_tile, apply_top1
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec
from scripts.plot_spatial_chunk import write_wikitext_table
from scripts.spatial_study import sweep_run


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

    def test_top1_keeps_farthest_token(self):
        source = _chunk()
        out = apply_top1(source, layer_idx=0, target="k", chunk=8)
        mean = source[0, 0, :8].mean(dim=0)
        self.assertTrue(torch.equal(out[0, 0, 3], source[0, 0, 3]))
        for index in (0, 1, 2, 4, 5, 6, 7):
            self.assertTrue(torch.allclose(out[0, 0, index], mean))
        self.assertTrue(torch.equal(out[0, 0, 8:], source[0, 0, 8:]))

    def test_feature_keeps_the_winning_coordinate(self):
        source = _chunk()
        out = apply_feature(source, layer_idx=0, target="k", chunk=8)
        mean = source[0, 0, :8].mean(dim=0)
        self.assertTrue(torch.allclose(out[0, 0, 3], source[0, 0, 3]))
        for index in (0, 1, 2, 4, 5, 6, 7):
            self.assertTrue(torch.allclose(out[0, 0, index], mean))

    def test_tile_keeps_the_winning_tile(self):
        source = _chunk()
        out = apply_tile(source, layer_idx=0, target="k", chunk=8, tile=8)
        mean = source[0, 0, :8].mean(dim=0)
        self.assertTrue(torch.allclose(out[0, 0, 3], source[0, 0, 3]))
        for index in (0, 1, 2, 4, 5, 6, 7):
            self.assertTrue(torch.allclose(out[0, 0, index], mean))

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


class PreRopeHookTests(unittest.TestCase):
    def test_inverse_then_forward_restores_keys(self):
        rope = RopeTables(rope_theta=500000.0, head_dim=16)
        keys = torch.randn(1, 2, 6, 16)
        positions = torch.arange(6)
        cos, sin = rope.cos_sin(positions, keys.dtype)
        rotated = apply_rope(keys, cos, sin, inverse=False)
        restored = apply_rope(rotated, cos, sin, inverse=True)
        self.assertTrue(torch.allclose(restored, keys, atol=1e-5))

    def test_pre_rope_hook_compresses_in_the_unrotated_space(self):
        from transformers.cache_utils import Cache, DynamicCache

        rope = RopeTables(rope_theta=10000.0, head_dim=4)
        spec = parse_kv_spec(spatial_pool(k_layers="all", v_layers=[], chunk=4, pre_rope=True))
        raw = torch.ones(1, 1, 4, 4)
        raw[0, 0, 0] = 5
        positions = torch.arange(4)
        cos, sin = rope.cos_sin(positions, raw.dtype)
        rotated = apply_rope(raw, cos, sin, inverse=False)
        uninstall = patch_cache_update(spec, rope)
        try:
            cache = DynamicCache()
            stored, values = Cache.update(cache, rotated, rotated.clone(), 0)
        finally:
            uninstall()
        pooled = apply_pool(raw, layer_idx=0, target="k", chunk=4)
        expected = apply_rope(pooled, cos, sin, inverse=False)
        self.assertTrue(torch.allclose(stored, expected, atol=1e-5))
        self.assertTrue(torch.allclose(values, rotated, atol=1e-5))

    def test_post_rope_flag_skips_the_inverse(self):
        from transformers.cache_utils import Cache, DynamicCache

        rope = RopeTables(rope_theta=10000.0, head_dim=4)
        spec = parse_kv_spec(spatial_pool(k_layers="all", v_layers=[], chunk=4, pre_rope=False))
        rotated = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        uninstall = patch_cache_update(spec, rope)
        try:
            cache = DynamicCache()
            stored, _values = Cache.update(cache, rotated.clone(), rotated.clone(), 0)
        finally:
            uninstall()
        expected = apply_pool(rotated, layer_idx=0, target="k", chunk=4)
        self.assertTrue(torch.allclose(stored, expected, atol=1e-5))


class SpatialRunTests(unittest.TestCase):
    def test_sweep_covers_every_method_slot_once(self):
        configurations = sweep_run()["configurations"]
        names = [item["name"] for item in configurations]
        self.assertEqual(names[0], "llama31_dense")
        self.assertEqual(names.count("llama31_dense"), 1)
        self.assertEqual(len(names), 1 + 4 * 32 * 2)
        self.assertEqual(configurations[0]["tasks"], ["wikitext"])
        pool_key = next(item for item in configurations if item["name"] == "llama31_spatial_pool_k00_p100")
        self.assertTrue(pool_key["kv"]["pipeline"][0]["pre_rope"])
        self.assertEqual(pool_key["kv"]["pipeline"][0]["k_layers"], [0])

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


class SpatialTableTests(unittest.TestCase):
    def test_table_uses_dense_baseline_and_stored_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dense = folder / "dense.json"
            dense.write_text(json.dumps({"results": {"wikitext": {"word_perplexity,none": 10.0}}}))
            for method in ("spatial_pool", "spatial_top1", "spatial_tile", "spatial_feature"):
                for target in ("k", "v", "both"):
                    _dump(folder / f"llama31_{method}_{target}.json", 11.0)
            _dump(folder / "llama31_spatial_pool_k_postrope.json", 12.0)
            _dump(folder / "llama31_spatial_top1_k_postrope.json", 10.5)
            table = write_wikitext_table(folder, dense, folder / "figures").read_text()
        self.assertIn("Baseline", table)
        self.assertIn("10.00", table)
        self.assertIn("1/8", table)
        self.assertIn("2/8", table)
        self.assertIn("post-RoPE", table)
        self.assertIn("20.00%", table)
        self.assertIn("5.00%", table)


def _dump(path: Path, perplexity: float) -> None:
    path.write_text(json.dumps({"results": {"wikitext": {"word_perplexity,none": perplexity}}}))


if __name__ == "__main__":
    unittest.main()
