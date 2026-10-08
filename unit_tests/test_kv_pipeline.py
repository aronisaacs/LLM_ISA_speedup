"""Unit tests for kv_compress: spec parsing, identity install, and KV methods.

Does not load a full LLM. Covers JSON validation, Cache.update wrapping,
sparsify 4:8, checksparse L1 tiles, per-scalar vector_compress, and
RoPE-paired keys.
"""

import unittest

import torch

from compression_topics.vector.algorithms.vector_compress import (
    apply as vector_apply,
    disable_zero_run_profile,
    enable_zero_run_profile,
    take_zero_run_profile,
    zero_run_count,
)
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

    def test_ratio_is_not_configurable(self):
        with self.assertRaisesRegex(ValueError, "unknown options"):
            parse_kv_spec(
                {"pipeline": [{"method": "sparsify_nm", "m": 6, "k_layers": "all", "v_layers": "all"}]}
            )

    def test_skips_layers_not_selected(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "sparsify_nm",
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


class VectorCompressPairTests(unittest.TestCase):
    def test_keys_drop_the_weakest_rope_pairs(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "vector_compress_pair",
                        "prune_pct": 50,
                        "k_layers": "all",
                        "v_layers": [],
                    }
                ]
            }
        )
        # Half is 2. Pairs are (0.2, 0.4) and (5, 0.1). The first pair is weaker.
        # Adjacent grouping would have kept 0.2 with 5 and dropped the other pair.
        key = torch.tensor([0.2, 5.0, 0.4, 0.1]).reshape(1, 1, 1, 4)
        out_key, out_value = compress_kv(key, key.clone(), layer_idx=0, spec=spec)
        self.assertTrue(torch.equal(out_key.reshape(4), torch.tensor([0.0, 5.0, 0.0, 0.1])))
        self.assertTrue(torch.equal(out_value, key))

    def test_values_match_scalar_prune(self):
        spec = parse_kv_spec(
            {
                "pipeline": [
                    {
                        "method": "vector_compress_pair",
                        "prune_pct": 50,
                        "k_layers": [],
                        "v_layers": "all",
                    }
                ]
            }
        )
        value = torch.tensor([0.2, 5.0, 0.4, 0.1]).reshape(1, 1, 1, 4)
        out_key, out_value = compress_kv(value.clone(), value, layer_idx=0, spec=spec)
        expected = vector_apply(value, layer_idx=0, target="v", prune_pct=50)
        self.assertTrue(torch.equal(out_key, value))
        self.assertTrue(torch.equal(out_value, expected))
        self.assertTrue(torch.equal(out_value.reshape(4), torch.tensor([0.0, 5.0, 0.4, 0.0])))

    def test_value_prune_does_not_record_zero_runs(self):
        from compression_topics.vector.algorithms.vector_compress_pair import apply as pair_apply

        value = torch.tensor([0.1, -0.9, 0.4, 2.0]).reshape(1, 1, 1, 4)
        enable_zero_run_profile()
        try:
            out = pair_apply(value, layer_idx=0, target="v", prune_pct=50)
            stats = take_zero_run_profile()
        finally:
            disable_zero_run_profile()
        self.assertTrue(torch.equal(out.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))
        self.assertEqual(stats["vectors"], 0)

    def test_zero_percent_is_identity_and_full_prune_is_zero(self):
        key = torch.tensor([1.0, -2.0, 0.5, 0.25]).reshape(1, 1, 1, 4)
        kept = parse_kv_spec(
            {"pipeline": [{"method": "vector_compress_pair", "prune_pct": 0, "k_layers": "all", "v_layers": "all"}]}
        )
        cleared = parse_kv_spec(
            {"pipeline": [{"method": "vector_compress_pair", "prune_pct": 100, "k_layers": "all", "v_layers": "all"}]}
        )
        kept_key, kept_value = compress_kv(key, key.clone(), layer_idx=0, spec=kept)
        cleared_key, cleared_value = compress_kv(key, key.clone(), layer_idx=0, spec=cleared)
        self.assertTrue(torch.equal(kept_key, key))
        self.assertTrue(torch.equal(kept_value, key))
        self.assertTrue(torch.equal(cleared_key, torch.zeros_like(key)))
        self.assertTrue(torch.equal(cleared_value, torch.zeros_like(key)))

    def test_pair_mask_profile_counts_dropped_pairs_once(self):
        from compression_topics.vector.algorithms.vector_compress_pair import (
            apply as pair_apply,
            disable_pair_mask_profile,
            enable_pair_mask_profile,
            random_pair_mask_runs,
            take_pair_mask_profile,
        )

        # Half is 4. Pairs 0 and 1 are the weak ones, so the mask is one run of two.
        # The stored key repeats that pattern in each half and would count two runs.
        key = torch.tensor([0.1, 0.1, 3.0, 3.0, 0.1, 0.1, 3.0, 3.0]).reshape(1, 1, 1, 8)
        enable_pair_mask_profile()
        try:
            out = pair_apply(key, layer_idx=0, target="k", prune_pct=50)
            stats = take_pair_mask_profile()
        finally:
            disable_pair_mask_profile()
        self.assertTrue(torch.equal(out.reshape(8), torch.tensor([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0])))
        self.assertEqual(stats["runs"], 1)
        self.assertEqual(stats["vectors"], 1)
        self.assertEqual(stats["mask_length"], 4)
        self.assertEqual(stats["mean"], 1.0)
        self.assertEqual(
            [random_pair_mask_runs(64, pct) for pct in (10, 20, 30, 40, 50, 60)],
            [5.53125, 9.9375, 13.65625, 15.625, 16.5, 16.03125],
        )

    def test_pair_mask_profile_stays_off_until_enabled(self):
        from compression_topics.vector.algorithms.vector_compress_pair import (
            apply as pair_apply,
            disable_pair_mask_profile,
            take_pair_mask_profile,
        )

        key = torch.tensor([0.1, 0.1, 3.0, 3.0, 0.1, 0.1, 3.0, 3.0]).reshape(1, 1, 1, 8)
        try:
            pair_apply(key, layer_idx=0, target="k", prune_pct=50)
            self.assertEqual(take_pair_mask_profile()["vectors"], 0)
        finally:
            disable_pair_mask_profile()

    def test_odd_feature_count_is_rejected_for_keys_only(self):
        odd = torch.tensor([1.0, 0.2, 3.0]).reshape(1, 1, 1, 3)
        keys = parse_kv_spec(
            {"pipeline": [{"method": "vector_compress_pair", "prune_pct": 50, "k_layers": "all", "v_layers": []}]}
        )
        values = parse_kv_spec(
            {"pipeline": [{"method": "vector_compress_pair", "prune_pct": 50, "k_layers": [], "v_layers": "all"}]}
        )
        with self.assertRaises(ValueError):
            compress_kv(odd, odd.clone(), layer_idx=0, spec=keys)
        out_key, out_value = compress_kv(odd.clone(), odd, layer_idx=0, spec=values)
        self.assertTrue(torch.equal(out_key, odd))
        self.assertTrue(torch.equal(out_value.reshape(3), torch.tensor([1.0, 0.0, 3.0])))


class ZeroRunProfileTests(unittest.TestCase):
    def tearDown(self):
        disable_zero_run_profile()

    def test_zero_run_count_known_patterns(self):
        self.assertEqual(zero_run_count(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])), 2)
        self.assertEqual(zero_run_count(torch.tensor([1.0, 2.0, 3.0])), 0)
        self.assertEqual(zero_run_count(torch.tensor([0.0, 0.0, 0.0])), 1)
        rows = torch.zeros(2, 3)
        self.assertEqual(zero_run_count(rows), 2)

    def test_profiled_prune_records_runs_without_changing_values(self):
        key = torch.tensor([0.1, -0.9, 0.4, 2.0]).reshape(1, 1, 1, 4)
        enable_zero_run_profile()
        out = vector_apply(key, layer_idx=0, target="k", prune_pct=50)
        stats = take_zero_run_profile()
        self.assertTrue(torch.equal(out.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))
        self.assertEqual(stats["runs"], 2)
        self.assertEqual(stats["vectors"], 1)
        self.assertEqual(stats["mean"], 2.0)
        again = vector_apply(key, layer_idx=0, target="k", prune_pct=50)
        self.assertTrue(torch.equal(again.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))
        fresh = take_zero_run_profile()
        self.assertEqual(fresh["runs"], 2)
        self.assertEqual(fresh["vectors"], 1)

    def test_profile_off_does_not_record(self):
        key = torch.tensor([0.1, -0.9, 0.4, 2.0]).reshape(1, 1, 1, 4)
        out = vector_apply(key, layer_idx=0, target="k", prune_pct=50)
        self.assertTrue(torch.equal(out.reshape(4), torch.tensor([0.0, -0.9, 0.0, 2.0])))
        self.assertEqual(take_zero_run_profile()["vectors"], 0)


if __name__ == "__main__":
    unittest.main()
