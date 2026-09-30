"""Adjacent-pair residual pooling. No model load."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import pair_pool, residual_pool
from compression_topics.spatial.algorithms.pair_pool import apply as apply_pair
from compression_topics.spatial.algorithms.residual_pool import apply, close_pairs
from compression_topics.spatial.scripts.residual_study import _kept_rows
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.apply import kv_for_slot, parse_singleton
from engine.layer_select.levels import mean_compression
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots


def _pair() -> torch.Tensor:
    # [batch, heads, seq=2, dim=4]. |delta| ranks feature 0, then 2, then 3, then 1.
    tokens = torch.tensor([5.0, 1.0, 0.4, 0.2, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    return tokens.reshape(1, 1, 2, 4)


class ResidualRewriteTests(unittest.TestCase):
    def test_full_residual_is_lossless_without_rope(self):
        source = torch.randn(1, 2, 3, 4)
        out = apply(source, layer_idx=0, target="k", prune_pct=0, rope=False)
        self.assertTrue(torch.allclose(out, source, atol=1e-5))
        self.assertTrue(torch.equal(out[:, :, 2], source[:, :, 2]))

    def test_full_residual_is_lossless_with_rope(self):
        tables = RopeTables(rope_theta=10000.0, head_dim=4)
        source = torch.randn(1, 2, 4, 4)
        out = apply(source, layer_idx=0, target="k", prune_pct=0, rope=True, rope_tables=tables)
        self.assertTrue(torch.allclose(out, source, atol=1e-5))

    def test_rope_flag_survives_next_to_the_tables(self):
        tables = RopeTables(rope_theta=10000.0, head_dim=4)
        source = torch.randn(1, 1, 2, 4)
        values = source.clone()
        spec = parse_kv_spec(residual_pool(prune_pct=0, rope=True))
        out_key, out_value = compress_kv(source, values, layer_idx=0, spec=spec, rope=tables)
        self.assertTrue(torch.allclose(out_key, source, atol=1e-5))
        self.assertTrue(torch.allclose(out_value, values, atol=1e-5))
        plain = parse_kv_spec(residual_pool(prune_pct=100, rope=False))
        plain_key, _plain_value = compress_kv(source, values, layer_idx=0, spec=plain, rope=tables)
        aligned = parse_kv_spec(residual_pool(prune_pct=100, rope=True))
        aligned_key, aligned_value = compress_kv(source.clone(), values.clone(), layer_idx=0, spec=aligned, rope=tables)
        self.assertFalse(torch.allclose(plain_key, aligned_key, atol=1e-5))
        self.assertTrue(torch.allclose(aligned_value, close_pairs(values, prune_pct=100, rope=False, rope_tables=None)))

    def test_one_step_rotation_is_what_gets_undone(self):
        tables = RopeTables(rope_theta=10000.0, head_dim=4)
        source = _pair()
        out = apply(source, layer_idx=0, target="k", prune_pct=100, rope=True, rope_tables=tables)
        first = source[:, :, :1]
        second = source[:, :, 1:]
        cos, sin = tables.cos_sin(torch.ones(1), source.dtype)
        aligned = apply_rope(second, cos, sin, inverse=True)
        mean = (first + aligned) / 2
        restored_second = apply_rope(mean, cos, sin, inverse=False)
        self.assertTrue(torch.allclose(out[:, :, :1], mean, atol=1e-5))
        self.assertTrue(torch.allclose(out[:, :, 1:], restored_second, atol=1e-5))

    def test_prune_keeps_the_largest_residual_features(self):
        source = _pair()
        out = apply(source, layer_idx=0, target="v", prune_pct=50, rope=False)
        # delta = [2, 0, 0.2, 0.1]. 50% drops the two weakest, features 1 and 3.
        self.assertTrue(torch.allclose(out[0, 0, 0], torch.tensor([5.0, 1.0, 0.4, 0.1]), atol=1e-5))
        self.assertTrue(torch.allclose(out[0, 0, 1], torch.tensor([1.0, 1.0, 0.0, 0.1]), atol=1e-5))

    def test_odd_tail_and_odd_start_stay_exact(self):
        source = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        out = apply(source, layer_idx=0, target="k", prune_pct=100, rope=False)
        self.assertTrue(torch.equal(out[:, :, 2], source[:, :, 2]))
        deferred = apply(source[:, :, :2], layer_idx=0, target="k", prune_pct=100, seq_start=1)
        self.assertTrue(torch.equal(deferred, source[:, :, :2]))

    def test_pair_pool_matches_full_residual_prune(self):
        source = torch.randn(1, 1, 5, 4)
        pooled = apply_pair(source, layer_idx=0, target="k")
        residual = apply(source, layer_idx=0, target="k", prune_pct=100, rope=False)
        self.assertTrue(torch.equal(pooled, residual))


class ResidualCacheTests(unittest.TestCase):
    def test_odd_prefill_closes_on_the_next_token(self):
        from transformers.cache_utils import DynamicCache

        from engine.kv_compress.cache import patch_cache_update

        spec = parse_kv_spec(residual_pool(k_layers="all", v_layers="all", prune_pct=50, rope=False))
        keys = _pair()
        values = torch.tensor([4.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.1], dtype=torch.float32).reshape(1, 1, 2, 4)
        uninstall = patch_cache_update(spec, RopeTables(rope_theta=10000.0, head_dim=4))
        try:
            cache = DynamicCache()
            stored_k, stored_v = cache.update(keys[:, :, :1], values[:, :, :1], 0)
            self.assertTrue(torch.equal(stored_k, keys[:, :, :1]))
            self.assertTrue(torch.equal(stored_v, values[:, :, :1]))
            stored_k, stored_v = cache.update(keys[:, :, 1:], values[:, :, 1:], 0)
            self.assertTrue(torch.allclose(stored_k, close_pairs(keys, prune_pct=50, rope=False, rope_tables=None)))
            self.assertTrue(torch.allclose(stored_v, close_pairs(values, prune_pct=50, rope=False, rope_tables=None)))
            tail_k = torch.tensor([9.0, 8.0, 7.0, 6.0]).view(1, 1, 1, 4)
            tail_v = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 1, 4)
            stored_k, stored_v = cache.update(tail_k, tail_v, 0)
            self.assertTrue(torch.allclose(stored_k[:, :, :2], close_pairs(keys, prune_pct=50, rope=False, rope_tables=None)))
            self.assertTrue(torch.equal(stored_k[:, :, 2], tail_k[:, :, 0]))
            self.assertTrue(torch.equal(stored_v[:, :, 2], tail_v[:, :, 0]))
        finally:
            uninstall()

    def test_rope_close_uses_the_tables_and_values_do_not(self):
        from transformers.cache_utils import DynamicCache

        from engine.kv_compress.cache import patch_cache_update

        tables = RopeTables(rope_theta=10000.0, head_dim=4)
        spec = parse_kv_spec(residual_pool(k_layers=[0], v_layers="all", prune_pct=100, rope=True))
        keys = _pair()
        values = keys.clone()
        uninstall = patch_cache_update(spec, tables)
        try:
            cache = DynamicCache()
            cache.update(keys[:, :, :1], values[:, :, :1], 0)
            stored_k, stored_v = cache.update(keys[:, :, 1:], values[:, :, 1:], 0)
        finally:
            uninstall()
        self.assertTrue(torch.allclose(stored_k, close_pairs(keys, prune_pct=100, rope=True, rope_tables=tables)))
        self.assertTrue(torch.allclose(stored_v, close_pairs(values, prune_pct=100, rope=False, rope_tables=None)))
        self.assertFalse(torch.allclose(stored_k, stored_v, atol=1e-5))

    def test_unselected_keys_stay_exact_when_values_pool(self):
        from transformers.cache_utils import DynamicCache

        from engine.kv_compress.cache import patch_cache_update

        spec = parse_kv_spec(pair_pool(k_layers=[], v_layers="all"))
        keys = _pair()
        values = _pair()
        uninstall = patch_cache_update(spec)
        try:
            cache = DynamicCache()
            cache.update(keys[:, :, :1], values[:, :, :1], 0)
            stored_k, stored_v = cache.update(keys[:, :, 1:], values[:, :, 1:], 0)
        finally:
            uninstall()
        self.assertTrue(torch.equal(stored_k, keys))
        self.assertTrue(torch.allclose(stored_v, close_pairs(values, prune_pct=100, rope=False, rope_tables=None)))


class ResidualRunTests(unittest.TestCase):
    def test_rungs_charge_the_mean_and_the_residual(self):
        levels = [rung.level for rung in rungs_for("residual_pool")]
        fractions = [rung.fraction for rung in rungs_for("residual_pool")]
        self.assertEqual(levels, [25, 50, 75, 100])
        self.assertEqual(fractions, [0.125, 0.25, 0.375, 0.5])
        pool = rungs_for("pair_pool")
        self.assertEqual(pool[0].level, 100)
        self.assertEqual(pool[0].fraction, 0.5)
        slots = all_slots(1)
        full = {slot: 100 for slot in slots}
        self.assertAlmostEqual(mean_compression(full, len(slots), rungs_for("residual_pool")), 0.5)
        self.assertAlmostEqual(mean_compression(full, len(slots), rungs_for("pair_pool")), 0.5)

    def test_slot_level_keeps_the_rope_flag(self):
        kv = kv_for_slot(residual_pool(prune_pct=25, rope=True), Slot(3, "k"), level=75)
        step = kv["pipeline"][0]
        self.assertEqual(step["prune_pct"], 75)
        self.assertIs(step["rope"], True)
        self.assertEqual(step["k_layers"], [3])
        self.assertEqual(parse_singleton(kv), (Slot(3, "k"), 75))
        pooled = kv_for_slot(pair_pool(), Slot(1, "v"), level=100)
        self.assertNotIn("prune_pct", pooled["pipeline"][0])
        self.assertEqual(parse_singleton(pooled), (Slot(1, "v"), 100))

    def test_score_filter_separates_the_two_residual_sweeps(self):
        def row(rope):
            return ScoreRow(
                slot=Slot(0, "k"),
                level=25,
                ppl=1.0,
                delta=0.1,
                path="",
                kv={"pipeline": [{"method": "residual_pool", "prune_pct": 25, "rope": rope}]},
            )

        rows = [row(False), row(True)]
        self.assertEqual(_kept_rows(rows, False)[0].kv["pipeline"][0]["rope"], False)
        self.assertEqual(_kept_rows(rows, True)[0].kv["pipeline"][0]["rope"], True)
        self.assertEqual(len(_kept_rows(rows, None)), 2)


if __name__ == "__main__":
    unittest.main()
