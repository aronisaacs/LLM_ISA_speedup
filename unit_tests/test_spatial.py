"""Chunk rewrites, inverse RoPE, and the 14-config WikiText run. No model load."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

import json
import tempfile

from catalog.compressions import spatial_pool, spatial_top1
from eval_runner.load_run import load_run
from kv_compress.cache import patch_cache_update
from kv_compress.methods.spatial import apply_feature, apply_pool, apply_tile, apply_top1
from kv_compress.rope import RopeTables, apply_rope
from kv_compress.spec import parse_kv_spec
from scripts.plot_spatial_chunk import write_wikitext_table

ROOT = Path(__file__).resolve().parents[1]


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
        out = apply_top1(source, layer_idx=0, target="v", chunk=8)
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
    def test_run_has_fourteen_wikitext_configs(self):
        loaded = load_run(ROOT / "runs" / "spatial_chunk.py")
        names = [configuration["name"] for configuration in loaded["configurations"]]
        self.assertEqual(len(names), 14)
        self.assertEqual(names[0], "llama31_spatial_pool_k")
        self.assertEqual(names[-2], "llama31_spatial_pool_k_postrope")
        self.assertEqual(names[-1], "llama31_spatial_top1_k_postrope")
        self.assertEqual(loaded["configurations"][0]["tasks"], ["wikitext"])
        self.assertTrue(loaded["configurations"][0]["kv"]["pipeline"][0]["pre_rope"])
        post = loaded["configurations"][-1]["kv"]["pipeline"][0]
        self.assertNotIn("pre_rope", post)
        self.assertEqual(post["method"], "spatial_top1")
        self.assertEqual(loaded["configurations"][0]["output_path"], "results/spatial_chunk/llama31_spatial_pool_k.json")
        keys_only = spatial_top1(k_layers="all", v_layers=[])
        self.assertEqual(keys_only["pipeline"][0]["v_layers"], [])


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
