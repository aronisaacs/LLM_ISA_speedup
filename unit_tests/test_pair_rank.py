"""Ranked pair merging (a fixed share of the most similar pairs per layer). No model load."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from catalog.compressions import pair_rank, pair_rank_residual
from compression_topics.spatial.algorithms import pair_gate, pair_rank as pair_rank_module
from compression_topics.spatial.algorithms.pair_rank import apply, merge_top_pairs
from compression_topics.spatial.scripts import pair_rank_study
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.apply import kv_for_assignment, kv_for_slot, parse_singleton
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import mean_compression
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots


def _graded(pairs: int = 8, heads: int = 2, features: int = 16) -> torch.Tensor:
    """Pair p of every head differs by a size that grows with p, so similarity is easy to rank."""
    torch.manual_seed(0)
    first = torch.randn(1, heads, pairs, features) + 4.0
    step = torch.arange(pairs, dtype=torch.float32).reshape(1, 1, pairs, 1) + 1.0
    second = first + 0.05 * step * torch.randn(1, heads, pairs, features)
    x = torch.empty(1, heads, 2 * pairs, features)
    x[..., 0::2, :] = first
    x[..., 1::2, :] = second
    return x


class MergeTopPairsTests(unittest.TestCase):
    def test_merges_the_requested_share_of_pairs(self):
        x = _graded()
        for pct, merged in ((0, 0), (25, 4), (50, 8), (75, 12), (100, 16)):
            pair_gate.reset_stats()
            merge_top_pairs(x, pct=pct, keep_pct=0, target="k")
            stats = pair_gate.pop_stats()["k"]
            self.assertEqual((stats["pairs"], stats["merged"]), (16, merged))

    def test_the_most_similar_pairs_merge_first(self):
        x = _graded()
        out = merge_top_pairs(x, pct=50, keep_pct=0)
        changed = (out != x).any(dim=-1)[..., 0::2]  # [1, heads, pairs]
        self.assertTrue(changed[..., :4].all())
        self.assertFalse(changed[..., 4:].any())
        mean = (x[..., 0::2, :] + x[..., 1::2, :]) / 2
        self.assertTrue(torch.allclose(out[..., 0::2, :][..., :4, :], mean[..., :4, :]))
        self.assertTrue(torch.allclose(out[..., 1::2, :][..., :4, :], mean[..., :4, :]))

    def test_ranking_is_across_heads(self):
        x = _graded()
        x[:, 1, 0::2, :] += 2.0
        x[:, 1, 1::2, :] -= 2.0  # head 1 pairs now differ by 4 in every feature
        out = merge_top_pairs(x, pct=50, keep_pct=0)
        merged = (out != x).any(dim=-1)[..., 0::2]
        self.assertEqual(int(merged[:, 0].sum()), 8)  # every head 0 pair is closer than any head 1 pair
        self.assertEqual(int(merged[:, 1].sum()), 0)

    def test_residual_changes_the_order_and_the_error(self):
        torch.manual_seed(1)
        a = torch.randn(1, 1, 1, 16) + 3.0
        spiky = a.clone()
        spiky[..., 5] += 6.0  # one big difference, all else identical
        mild = a + 0.4 * torch.randn_like(a)
        x = torch.cat([a, spiky, a, mild], dim=-2)  # pair 0 is spiky, pair 1 is mildly noisy
        plain = merge_top_pairs(x, pct=50, keep_pct=0)
        self.assertTrue(torch.equal(plain[..., :2, :], x[..., :2, :]))  # the mild pair merged first
        self.assertFalse(torch.equal(plain[..., 2:, :], x[..., 2:, :]))
        residual = merge_top_pairs(x, pct=50, keep_pct=7)  # one feature kept exactly
        self.assertTrue(torch.allclose(residual[..., :2, :], x[..., :2, :], atol=1e-5))  # the spike is restored
        self.assertTrue(torch.equal(residual[..., 2:, :], x[..., 2:, :]))

    def test_tail_token_stays_exact(self):
        x = _graded()
        odd = torch.cat([x, x[..., :1, :]], dim=-2)
        out = merge_top_pairs(odd, pct=100, keep_pct=0)
        self.assertTrue(torch.equal(out[..., -1:, :], odd[..., -1:, :]))

    def test_bytes_follow_the_pairs_merged(self):
        x = _graded()
        pair_gate.reset_stats()
        merge_top_pairs(x, pct=50, keep_pct=0, target="k")
        self.assertAlmostEqual(pair_gate.bytes_vs_dense(pair_gate.pop_stats()["k"]), 0.75)
        merge_top_pairs(x, pct=100, keep_pct=25, target="k")
        self.assertAlmostEqual(pair_gate.bytes_vs_dense(pair_gate.pop_stats()["k"]), 0.65625)


class ApplyTests(unittest.TestCase):
    def test_a_later_chunk_and_a_padded_batch_are_handled(self):
        x = _graded()
        self.assertTrue(torch.equal(apply(x, layer_idx=0, target="k", pct=100, seq_start=4), x))
        with self.assertRaises(ValueError):
            apply(x.expand(2, -1, -1, -1), layer_idx=0, target="k", pct=50)

    def test_residual_variant_keeps_a_quarter_of_the_difference(self):
        x = _graded()
        out = pair_rank_module.apply_residual(x, layer_idx=0, target="k", pct=100)
        self.assertTrue(torch.equal(out, merge_top_pairs(x, pct=100, keep_pct=25)))

    def test_rope_alignment_is_a_no_op_for_values_and_identical_pairs(self):
        x = _graded()
        out = apply(x, layer_idx=0, target="v", pct=50, rope=True)  # values ignore rope, so no tables needed
        self.assertTrue(torch.equal(out, merge_top_pairs(x, pct=50, keep_pct=0)))

    def test_rope_alignment_makes_a_rotated_copy_a_perfect_pair(self):
        from engine.kv_compress.rope import RopeTables, apply_rope

        tables = RopeTables(rope_theta=10000.0, head_dim=16)
        base = torch.randn(1, 2, 4, 16)
        keys = torch.empty(1, 2, 8, 16)
        for slot in range(8):  # the same vector at every position, stored after RoPE
            cos, sin = tables.cos_sin(torch.tensor([float(slot)]), base.dtype)
            keys[..., slot : slot + 1, :] = apply_rope(base[..., slot // 2 : slot // 2 + 1, :], cos, sin, inverse=False)
        aligned = merge_top_pairs(keys, pct=100, keep_pct=0, rope=True, rope_tables=tables)
        self.assertTrue(torch.allclose(aligned, keys, atol=1e-5))
        plain = merge_top_pairs(keys, pct=100, keep_pct=0)
        self.assertFalse(torch.allclose(plain, keys, atol=1e-3))

    def test_method_registered_and_spec_parses(self):
        for kv in (pair_rank(pct=50), pair_rank_residual(pct=50, rope=True)):
            spec = parse_kv_spec(kv)
            self.assertEqual(len(spec.pipeline), 1)


class RunPlumbingTests(unittest.TestCase):
    def test_keys_only_slots(self):
        slots = all_slots(3, ("k",))
        self.assertEqual(slots, (Slot(0, "k"), Slot(1, "k"), Slot(2, "k")))
        with self.assertRaises(ValueError):
            all_slots(3, ())

    def test_rungs_charge_the_overhead(self):
        merge = rungs_for("pair_rank")
        self.assertEqual([rung.level for rung in merge], [25, 50, 75, 100])
        self.assertEqual([rung.fraction for rung in merge], [0.125, 0.25, 0.375, 0.5])
        residual = rungs_for("pair_rank_residual")
        self.assertAlmostEqual(residual[-1].fraction, 0.34375)
        self.assertAlmostEqual(residual[0].fraction, 0.34375 / 4)

    def test_slot_level_sets_pct_and_round_trips(self):
        kv = kv_for_slot(pair_rank(pct=25), Slot(5, "k"), level=75)
        step = kv["pipeline"][0]
        self.assertEqual((step["pct"], step["k_layers"], step["v_layers"]), (75, [5], []))
        self.assertEqual(parse_singleton(kv), (Slot(5, "k"), 75))
        parse_kv_spec(kv)

    def test_assignment_builds_one_step_per_level(self):
        kv = kv_for_assignment(pair_rank_residual(pct=25), {Slot(0, "k"): 50, Slot(1, "k"): 100, Slot(2, "k"): 50})
        steps = {step["pct"]: step for step in kv["pipeline"]}
        self.assertEqual(sorted(steps), [50, 100])
        self.assertEqual(steps[50]["k_layers"], [0, 2])
        parse_kv_spec(kv)

    def test_rank_fill_budget_is_a_share_of_the_key_bytes(self):
        rungs = rungs_for("pair_rank")
        slots = all_slots(4, ("k",))
        rows = [
            ScoreRow(slot=slot, level=rung.level, ppl=10.0 + 0.1 * (index + 1) * (rank + 1), delta=0.0, path="")
            for index, slot in enumerate(slots)
            for rank, rung in enumerate(rungs)
        ]
        assignment = rank_fill(rows, 4, 0.4, dense_ppl=10.0, rungs=rungs, targets=("k",))
        self.assertGreaterEqual(mean_compression(assignment, len(slots), rungs), 0.4)
        self.assertTrue(all(slot.target == "k" for slot in assignment))
        # the cheapest layer (smallest perplexity cost) climbs furthest
        self.assertGreaterEqual(assignment[Slot(0, "k")], assignment[Slot(3, "k")])


class StudyTests(unittest.TestCase):
    def test_sweep_is_the_key_slots_at_four_rungs(self):
        run = pair_rank_study.sweep_run("merge", Path("/tmp/sweep"))
        self.assertEqual(run["batch_size"], 1)
        self.assertEqual(len(run["configurations"]), 1 + 32 * 4)
        for configuration in run["configurations"][1:]:
            self.assertEqual(configuration["layer_slot"]["target"], "k")
            self.assertEqual(configuration["kv"]["pipeline"][0]["method"], "pair_rank")

    def test_task_run_has_dense_greedy_and_uniform(self):
        payloads = [
            {
                "tag": "merge_p10",
                "budget": 0.1,
                "compression": 0.125,
                "kv": kv_for_assignment(pair_rank(pct=25), {Slot(0, "k"): 100}),
            }
        ]
        run = pair_rank_study.task_run("merge", Path("/tmp/tasks"), payloads)
        names = [configuration["name"] for configuration in run["configurations"]]
        self.assertEqual(
            names,
            ["llama31_ceval_dense", "llama31_ceval_merge_p10"]
            + [f"llama31_ceval_merge_uniform_p{pct}" for pct in (25, 50, 75, 100)],
        )
        uniform = run["configurations"][-1]
        self.assertEqual(uniform["kv"]["pipeline"][0]["pct"], 100)
        self.assertEqual(uniform["kv"]["pipeline"][0]["k_layers"], "all")


if __name__ == "__main__":
    unittest.main()
