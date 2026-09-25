"""Hadamard codebook rewrite and the WikiText run. No model load."""

from __future__ import annotations

import math
import unittest

import torch

from catalog.compressions import qjl
from engine.kv_compress.methods.qjl import apply, centroids, hadamard, sign_diagonal
from engine.layer_select.apply import kv_for_assignment
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import next_level
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots
from scripts.qjl_study import sweep_run


class QjlRewriteTests(unittest.TestCase):
    def test_hadamard_is_its_own_inverse(self):
        values = torch.randn(2, 8)
        self.assertTrue(torch.allclose(hadamard(hadamard(values)), values, atol=1e-5))

    def test_one_bit_centroids_match_the_gaussian_mean(self):
        level = math.sqrt(2.0 / math.pi) / math.sqrt(4)
        self.assertTrue(torch.allclose(centroids(1, 4), torch.tensor([-level, level]), atol=1e-5))

    def test_rebuild_uses_the_rotated_codebook_and_fp16_norm(self):
        features = 8
        bits = 4
        token = torch.randn(features)
        rebuilt = apply(token.view(1, 1, 1, features), layer_idx=2, target="k", bits=bits).view(-1)
        norm = token.norm()
        stored = norm.to(dtype=torch.float16).float()
        signs = sign_diagonal(2, features)
        rotated = hadamard(token / norm * signs)
        levels = centroids(bits, features)
        snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
        expected = hadamard(snapped) * signs * stored
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-5))

    def test_values_use_the_same_rebuild(self):
        token = torch.randn(1, 1, 1, 8)
        keys = apply(token, layer_idx=1, target="k", bits=4)
        values = apply(token, layer_idx=1, target="v", bits=4)
        self.assertTrue(torch.equal(keys, values))

    def test_zero_token_stays_zero(self):
        token = torch.zeros(1, 1, 2, 4)
        self.assertTrue(torch.equal(apply(token, layer_idx=0, target="k", bits=4), token))


class QjlRunTests(unittest.TestCase):
    def test_sweep_scores_every_slot_at_each_bit_width(self):
        configurations = sweep_run()["configurations"]
        names = [item["name"] for item in configurations]
        self.assertEqual(names[0], "llama31_dense")
        self.assertEqual(len(names), 1 + 32 * 2 * 4)
        four = next(item for item in configurations if item["name"] == "llama31_qjl_k00_p4")
        one = next(item for item in configurations if item["name"] == "llama31_qjl_k00_p1")
        self.assertEqual(four["kv"]["pipeline"][0]["method"], "qjl")
        self.assertEqual(four["kv"]["pipeline"][0]["bits"], 4)
        self.assertEqual(one["kv"]["pipeline"][0]["bits"], 1)
        self.assertNotIn("pre_rope", four["kv"]["pipeline"][0])
        self.assertEqual(four["tasks"], ["wikitext"])

    def test_climber_walks_four_bits_down_to_one(self):
        widths = tuple(rung.level for rung in rungs_for("qjl"))
        self.assertEqual(widths, (4, 3, 2, 1))
        self.assertEqual([next_level(0, widths), next_level(4, widths), next_level(1, widths)], [4, 3, None])

    def test_assignment_can_mix_four_bits_and_one_bit(self):
        kv = kv_for_assignment(qjl(), {Slot(3, "k"): 4, Slot(7, "v"): 1})
        by_bits = {step["bits"]: step for step in kv["pipeline"]}
        self.assertEqual(by_bits[4]["k_layers"], [3])
        self.assertEqual(by_bits[4]["v_layers"], [])
        self.assertEqual(by_bits[1]["k_layers"], [])
        self.assertEqual(by_bits[1]["v_layers"], [7])

    def test_greedy_leaves_one_slot_at_four_bits_and_climbs_another_to_one(self):
        levels = tuple(rung.level for rung in rungs_for("qjl"))
        rows = []
        for slot in all_slots(1):
            for level in levels:
                rows.append(ScoreRow(slot=slot, level=level, ppl=60.0, delta=50.0, path=f"{slot.tag()}p{level}.json"))
        cheap = {
            (Slot(0, "v"), 4): 0.01,
            (Slot(0, "k"), 4): 0.05,
            (Slot(0, "k"), 3): 0.06,
            (Slot(0, "k"), 2): 0.07,
            (Slot(0, "k"), 1): 0.08,
        }
        for (slot, level), delta in cheap.items():
            rows = [row for row in rows if not (row.slot == slot and row.level == level)]
            rows.append(ScoreRow(slot=slot, level=level, ppl=10.0 + delta, delta=delta, path=f"{slot.tag()}p{level}.json"))
        chosen = rank_fill(rows, n_layers=1, budget=0.84, dense_ppl=10.0, rungs=rungs_for("qjl"))
        self.assertEqual(chosen, {Slot(0, "v"): 4, Slot(0, "k"): 1})


if __name__ == "__main__":
    unittest.main()
