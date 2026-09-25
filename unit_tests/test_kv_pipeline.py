"""Unit tests for kv_compress: spec parsing, identity install, and KV methods.

Does not load a full LLM. Covers JSON validation, Cache.update wrapping,
sparsify 4:8, checksparse L1 tiles, and per-scalar vector_compress.
"""

import unittest

import torch

from engine.kv_compress.cache import patch_cache_update
from engine.kv_compress.install import install
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.spec import parse_kv_spec


class ParseKvSpecTests(unittest.TestCase):
    def test_omitted_and_empty_are_identity(self):
        for value in (None, {}, {"pipeline": []}):
            spec = parse_kv_spec(value)
            self.assertTrue(spec.is_identity())
            self.assertEqual(spec.to_dict(), {"pipeline": []})

    def test_unknown_kv_key_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"backend": "ours", "pipeline": []})

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"pipeline": [{"method": "not_a_method", "k_layers": "all"}]})

    def test_layer_lists_and_all(self):
        from engine.kv_compress import methods as methods_module

        def _identity(tensor, **_kwargs):
            return tensor

        methods_module.METHODS["identity"] = _identity
        try:
            spec = parse_kv_spec(
                {
                    "pipeline": [
                        {"method": "identity", "k_layers": "all", "v_layers": [0, 2]}
                    ]
                }
            )
            step = spec.pipeline[0]
            self.assertEqual(step.k_layers, "all")
            self.assertEqual(step.v_layers, frozenset({0, 2}))
            self.assertEqual(
                spec.to_dict(),
                {
                    "pipeline": [
                        {"method": "identity", "k_layers": "all", "v_layers": [0, 2]}
                    ]
                },
            )
        finally:
            methods_module.METHODS.pop("identity", None)

    def test_negative_layer_is_rejected(self):
        from engine.kv_compress import methods as methods_module

        def _identity(tensor, **_kwargs):
            return tensor

        methods_module.METHODS["identity"] = _identity
        try:
            with self.assertRaises(ValueError):
                parse_kv_spec({"pipeline": [{"method": "identity", "k_layers": [-1]}]})
        finally:
            methods_module.METHODS.pop("identity", None)


class PipelineAndInstallTests(unittest.TestCase):
    def test_identity_pipeline_leaves_tensors_unchanged(self):
        spec = parse_kv_spec({"pipeline": []})
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        value = key + 1
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=spec)
        self.assertIs(out_key, key)
        self.assertIs(out_value, value)

    def test_install_identity_does_not_patch_cache(self):
        from transformers.cache_utils import Cache

        original = Cache.update
        uninstall = install(None, parse_kv_spec({"pipeline": []}))
        try:
            self.assertIs(Cache.update, original)
        finally:
            uninstall()
        self.assertIs(Cache.update, original)

    def test_cache_update_identity_patch_round_trip(self):
        from transformers.cache_utils import Cache, DynamicCache

        spec = parse_kv_spec({"pipeline": []})
        original = Cache.update
        uninstall = patch_cache_update(spec)
        try:
            self.assertIsNot(Cache.update, original)
            cache = DynamicCache()
            key = torch.randn(1, 2, 3, 4)
            value = torch.randn(1, 2, 3, 4)
            out_key, out_value = cache.update(key, value, 0)
            self.assertTrue(torch.equal(out_key, key))
            self.assertTrue(torch.equal(out_value, value))
        finally:
            uninstall()
        self.assertIs(Cache.update, original)


class SparsifyNmTests(unittest.TestCase):
    def test_keeps_m_largest_per_tile(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "sparsify_nm",
                        "n": 8,
                        "m": 4,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.tensor(
            [1.0, -9.0, 2.0, 3.0, -4.0, 8.0, 0.5, -0.1, 10.0, 1.0, -7.0, 0.0, 2.0, -3.0, 6.0, 0.2]
        ).reshape(1, 1, 1, 16)
        value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16)
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=spec)

        self.assertEqual(tuple(out_key.shape), tuple(key.shape))
        zeros_per_tile = (out_key.reshape(2, 8) == 0).sum(dim=-1)
        nonzero_per_tile = (out_key.reshape(2, 8) != 0).sum(dim=-1)
        self.assertTrue(torch.equal(zeros_per_tile, torch.tensor([4, 4])))
        self.assertTrue(torch.equal(nonzero_per_tile, torch.tensor([4, 4])))
        # Tile 0 magnitudes: 1,9,2,3,4,8,0.5,0.1 → keep 9,8,4,3 → values -9,8,-4,3
        self.assertTrue(
            torch.equal(out_key.reshape(2, 8)[0], torch.tensor([0.0, -9.0, 0.0, 3.0, -4.0, 8.0, 0.0, 0.0]))
        )

        value_tiles = out_value.reshape(2, 8)
        for tile, original in zip(value_tiles, value.reshape(2, 8)):
            self.assertEqual(int((tile != 0).sum()), 4)
            kept = original.abs().topk(4).indices
            self.assertTrue(torch.equal(tile[kept], original[kept]))
            dropped = torch.ones(8, dtype=torch.bool)
            dropped[kept] = False
            self.assertTrue(torch.all(tile[dropped] == 0))

    def test_skips_layers_not_selected(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "sparsify_nm",
                        "n": 8,
                        "m": 4,
                        "k_layers": [1],
                        "v_layers": [],
                    }
                ]
            }
        )
        key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        value = key + 1
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key, key))
        self.assertTrue(torch.equal(out_value, value))
        out_key, out_value = compress_kv(key, value, layer_idx=1, spec=spec)
        self.assertEqual(int((out_key == 0).sum()), 4)
        self.assertTrue(torch.equal(out_value, value))

    def test_install_patches_and_sparsifies_cache_update(self):
        from transformers.cache_utils import Cache, DynamicCache

        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "sparsify_nm",
                        "n": 8,
                        "m": 4,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        original = Cache.update
        uninstall = install(None, spec)
        try:
            self.assertIsNot(Cache.update, original)
            cache = DynamicCache()
            key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
            value = key.clone()
            out_key, out_value = cache.update(key, value, 0)
            self.assertEqual(int((out_key == 0).sum()), 4)
            self.assertEqual(int((out_value == 0).sum()), 4)
        finally:
            uninstall()
        self.assertIs(Cache.update, original)


class ChecksparseL1Tests(unittest.TestCase):
    def test_zeros_weakest_tiles_by_l1(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "checksparse_l1",
                        "tile": 8,
                        "prune_pct": 50,
                        "k_layers": "all",
                        "v_layers": [],
                    }
                ]
            }
        )
        # Tile 0 L1=8, tile 1 L1=0.8 → drop tile 1.
        key = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ).reshape(1, 1, 1, 16)
        value = key + 1
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key.reshape(16)[:8], key.reshape(16)[:8]))
        self.assertTrue(torch.equal(out_key.reshape(16)[8:], torch.zeros(8)))
        self.assertTrue(torch.equal(out_value, value))

    def test_prune_zero_is_identity(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "checksparse_l1",
                        "tile": 8,
                        "prune_pct": 0,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        out_key, _ = compress_kv(key, key, layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key, key))


class VectorCompressTests(unittest.TestCase):
    def test_zeros_scalars_below_threshold(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "vector_compress",
                        "threshold": 0.5,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.tensor([0.1, -0.9, 0.4, 2.0]).reshape(1, 1, 1, 4)
        out_key, _ = compress_kv(key, key.clone(), layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))

    def test_threshold_zero_is_identity(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "vector_compress",
                        "threshold": 0,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.tensor([0.0, 0.1, -2.0]).reshape(1, 1, 1, 3)
        out_key, _ = compress_kv(key, key, layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key, key))

    def test_prune_pct_zeros_weakest_scalars(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "vector_compress",
                        "prune_pct": 50,
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.tensor([0.1, -0.9, 0.4, 2.0]).reshape(1, 1, 1, 4)
        out_key, _ = compress_kv(key, key.clone(), layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))


class DynamicPrecisionTests(unittest.TestCase):
    def test_loud_tile_stays_full_quiet_tile_is_coarse(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "dynamic_precision",
                        "tile": 8,
                        "bits": [16, 4],
                        "pcts": [50, 50],
                        "k_layers": "all",
                        "v_layers": [],
                    }
                ]
            }
        )
        loud = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        quiet = torch.tensor([0.01, 0.03, -0.02, 0.09, 0.04, -0.07, 0.05, 0.20])
        key = torch.cat([loud, quiet]).reshape(1, 1, 1, 16)
        out_key, out_value = compress_kv(key, key.clone(), layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key.reshape(16)[:8], loud))
        self.assertFalse(torch.equal(out_key.reshape(16)[8:], quiet))
        self.assertTrue(torch.equal(out_value, key))

    def test_all_16bit_is_identity(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "dynamic_precision",
                        "tile": 8,
                        "bits": [16],
                        "pcts": [100],
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        out_key, _ = compress_kv(key, key, layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key, key))


if __name__ == "__main__":
    unittest.main()
