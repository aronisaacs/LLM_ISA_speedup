"""Pair pooling with a quantized residual. No model load."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import pair_pool, pair_quant
from compression_topics.spatial.algorithms.pair_pool import apply as apply_pair_pool
from compression_topics.spatial.algorithms.pair_quant import apply, close_pairs, quantize_residual
from engine.layer_select.apply import kv_for_assignment, kv_for_slot, parse_singleton
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import mean_compression, next_level
from engine.layer_select.rungs import MERGE_ONLY, rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.spec import parse_kv_spec


def _pair() -> torch.Tensor:
    # [batch, heads, seq=2, dim=4]. delta = [2, 0, 0.2, 0.1].
    tokens = torch.tensor([5.0, 1.0, 0.4, 0.2, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    return tokens.reshape(1, 1, 2, 4)


class PairQuantRewriteTests(unittest.TestCase):
    def test_zero_bits_is_pair_pool(self):
        source = torch.randn(1, 2, 5, 64)
        pooled = apply(source, layer_idx=0, target="k", bits=0)
        self.assertTrue(torch.equal(pooled, apply_pair_pool(source, layer_idx=0, target="k")))

    def test_eight_bits_is_nearly_lossless(self):
        source = torch.randn(1, 2, 4, 64)
        out = apply(source, layer_idx=0, target="v", bits=8)
        self.assertTrue(torch.allclose(out, source, atol=0.02))
        self.assertFalse(torch.equal(out, apply(source, layer_idx=0, target="v", bits=0)))

    def test_error_shrinks_as_bits_grow(self):
        torch.manual_seed(0)
        source = torch.randn(1, 4, 16, 128)
        errors = [
            (apply(source, layer_idx=0, target="v", bits=bits) - source).abs().mean().item()
            for bits in (0, 1, 2, 4, 8)
        ]
        self.assertEqual(errors, sorted(errors, reverse=True))

    def test_the_mean_is_stored_exactly_and_the_pair_is_symmetric(self):
        source = torch.randn(1, 2, 6, 64)
        out = apply(source, layer_idx=0, target="k", bits=2)
        mean = (out[..., 0::2, :] + out[..., 1::2, :]) / 2
        expected = (source[..., 0::2, :] + source[..., 1::2, :]) / 2
        self.assertTrue(torch.allclose(mean, expected, atol=1e-5))

    def test_two_bits_hold_four_levels_of_the_group_absmax(self):
        delta = torch.tensor([[3.0, -2.5, 0.4, -0.2]])
        out = quantize_residual(delta, 2, group=4)
        # absmax 3. Levels are -3, -1, 1, 3.
        self.assertTrue(torch.allclose(out, torch.tensor([[3.0, -3.0, 1.0, -1.0]])))
        groups = quantize_residual(torch.randn(1, 128), 2, group=32).reshape(4, 32)
        for row in groups:
            self.assertLessEqual(len(row.unique()), 4)

    def test_one_bit_keeps_the_sign_at_the_mean_magnitude(self):
        delta = torch.tensor([[3.0, -2.0, 1.0, -2.0]])
        out = quantize_residual(delta, 1, group=4)
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, -2.0, 2.0, -2.0]])))

    def test_groups_scale_independently_and_a_short_group_is_handled(self):
        delta = torch.tensor([[8.0, 4.0, 0.2, 0.1, 6.0]])
        out = quantize_residual(delta, 8, group=2)
        self.assertTrue(torch.allclose(out, delta, atol=0.05))
        self.assertTrue(torch.allclose(quantize_residual(delta, 8, group=4)[:, 4:], delta[:, 4:], atol=0.05))

    def test_odd_tail_and_odd_start_stay_exact(self):
        source = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        out = apply(source, layer_idx=0, target="k", bits=4)
        self.assertTrue(torch.equal(out[:, :, 2], source[:, :, 2]))
        deferred = apply(source[:, :, :2], layer_idx=0, target="k", bits=4, seq_start=1)
        self.assertTrue(torch.equal(deferred, source[:, :, :2]))

    def test_rejects_bad_settings(self):
        source = _pair()
        for bits in (3, 5, 16, -1):
            with self.assertRaises(ValueError):
                apply(source, layer_idx=0, target="k", bits=bits)
        with self.assertRaises(ValueError):
            apply(source, layer_idx=0, target="x", bits=4)
        with self.assertRaises(ValueError):
            apply(source, layer_idx=0, target="k", bits=4, group=0)

    def test_pipeline_selects_layers_for_keys_and_values(self):
        spec = parse_kv_spec(pair_quant(k_layers=[0], v_layers=[], bits=0))
        keys = _pair()
        values = _pair()
        out_k, out_v = compress_kv(keys, values, layer_idx=0, spec=spec)
        self.assertTrue(torch.allclose(out_k, close_pairs(keys, bits=0)))
        self.assertTrue(torch.equal(out_v, values))
        out_k, _out_v = compress_kv(keys, values, layer_idx=1, spec=spec)
        self.assertTrue(torch.equal(out_k, keys))


class PairQuantCacheTests(unittest.TestCase):
    def test_odd_prefill_closes_on_the_next_token(self):
        from transformers.cache_utils import DynamicCache

        from engine.kv_compress.cache import patch_cache_update

        spec = parse_kv_spec(pair_quant(k_layers="all", v_layers="all", bits=2, group=4))
        keys = _pair()
        values = torch.tensor([4.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.1], dtype=torch.float32).reshape(1, 1, 2, 4)
        uninstall = patch_cache_update(spec)
        try:
            cache = DynamicCache()
            stored_k, stored_v = cache.update(keys[:, :, :1], values[:, :, :1], 0)
            self.assertTrue(torch.equal(stored_k, keys[:, :, :1]))
            stored_k, stored_v = cache.update(keys[:, :, 1:], values[:, :, 1:], 0)
            self.assertTrue(torch.allclose(stored_k, close_pairs(keys, bits=2, group=4)))
            self.assertTrue(torch.allclose(stored_v, close_pairs(values, bits=2, group=4)))
            tail = torch.tensor([9.0, 8.0, 7.0, 6.0]).view(1, 1, 1, 4)
            stored_k, _stored_v = cache.update(tail, tail, 0)
            self.assertTrue(torch.allclose(stored_k[:, :, :2], close_pairs(keys, bits=2, group=4)))
            self.assertTrue(torch.equal(stored_k[:, :, 2], tail[:, :, 0]))
        finally:
            uninstall()

    def test_unselected_keys_stay_exact_when_values_close(self):
        from transformers.cache_utils import DynamicCache

        from engine.kv_compress.cache import patch_cache_update

        spec = parse_kv_spec(pair_quant(k_layers=[], v_layers="all", bits=0))
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
        self.assertTrue(torch.allclose(stored_v, close_pairs(values, bits=0)))


class PairQuantRunTests(unittest.TestCase):
    def test_rungs_charge_the_mean_and_the_residual_bits(self):
        rungs = rungs_for("pair_quant")
        self.assertEqual([rung.level for rung in rungs], [8, 4, 2, 1, MERGE_ONLY])
        self.assertEqual([rung.fraction for rung in rungs], [0.25, 0.375, 0.4375, 0.46875, 0.5])
        slots = all_slots(1)
        full = {slot: MERGE_ONLY for slot in slots}
        self.assertAlmostEqual(mean_compression(full, len(slots), rungs), 0.5)

    def test_levels_walk_mildest_first_and_merge_only_is_last(self):
        levels = tuple(rung.level for rung in rungs_for("pair_quant"))
        walked = [next_level(0, levels)]
        while walked[-1] is not None:
            walked.append(next_level(walked[-1], levels))
        self.assertEqual(walked, [8, 4, 2, 1, MERGE_ONLY, None])

    def test_slot_level_sets_the_bits_and_round_trips(self):
        kv = kv_for_slot(pair_quant(bits=8), Slot(3, "k"), level=2)
        step = kv["pipeline"][0]
        self.assertEqual(step["bits"], 2)
        self.assertEqual(step["k_layers"], [3])
        self.assertEqual(parse_singleton(kv), (Slot(3, "k"), 2))
        merged = kv_for_slot(pair_quant(bits=8), Slot(1, "v"), level=MERGE_ONLY)
        self.assertEqual(merged["pipeline"][0]["bits"], 0)
        self.assertEqual(parse_singleton(merged), (Slot(1, "v"), MERGE_ONLY))

    def test_assignment_builds_one_step_per_level(self):
        kv = kv_for_assignment(
            pair_quant(bits=8),
            {Slot(0, "k"): 4, Slot(1, "v"): MERGE_ONLY, Slot(2, "k"): 4},
        )
        steps = {step["bits"]: step for step in kv["pipeline"]}
        self.assertEqual(sorted(steps), [0, 4])
        self.assertEqual(steps[4]["k_layers"], [0, 2])
        self.assertEqual(steps[0]["v_layers"], [1])
        parse_kv_spec(kv)

    def test_rank_fill_reaches_the_full_budget_through_merge_only(self):
        rungs = rungs_for("pair_quant")
        slots = all_slots(2)
        rows = [
            ScoreRow(slot=slot, level=rung.level, ppl=10.0 + 0.1 * (index + 1) * (rank + 1), delta=0.0, path="")
            for index, slot in enumerate(slots)
            for rank, rung in enumerate(rungs)
        ]
        assignment = rank_fill(rows, 2, 0.5, dense_ppl=10.0, rungs=rungs)
        self.assertEqual(set(assignment.values()), {MERGE_ONLY})
        self.assertAlmostEqual(mean_compression(assignment, len(slots), rungs), 0.5)

    def test_pair_pool_still_has_its_own_single_rung(self):
        self.assertEqual(rungs_for("pair_pool")[0].level, 100)
        self.assertEqual(parse_singleton(kv_for_slot(pair_pool(), Slot(1, "v"), level=100)), (Slot(1, "v"), 100))


if __name__ == "__main__":
    unittest.main()
