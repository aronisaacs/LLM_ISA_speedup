"""Uniform int8/int4 rewrite and the WikiText run. No model load."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import quantize
from engine.kv_compress.methods.quantize import apply
from engine.layer_select.apply import kv_for_assignment
from engine.layer_select.levels import next_level
from engine.layer_select.rungs import rungs_for
from engine.layer_select.slots import Slot
from scripts.quantize_study import sweep_run


def _qdq(values: torch.Tensor, bits: int) -> torch.Tensor:
    max_q = (1 << (bits - 1)) - 1
    scale = values.abs().amax(dim=-1, keepdim=True)
    codes = (values / scale * max_q).round().clamp(-max_q, max_q)
    return codes * (scale / max_q)


class QuantizeRewriteTests(unittest.TestCase):
    def test_values_share_a_scale_across_features(self):
        tensor = torch.tensor([[[[8.0, 1.0, -0.5, 0.25], [0.4, -0.2, 0.1, 0.05]]]])
        rebuilt = apply(tensor, layer_idx=0, target="v", bits=4, group=4)
        expected = _qdq(tensor, 4)
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-5))

    def test_keys_share_a_scale_across_tokens(self):
        feature0 = torch.tensor([4.0, -2.0, 1.0, 0.5])
        feature1 = torch.tensor([0.4, -0.2, 0.1, 0.05])
        tensor = torch.stack((feature0, feature1), dim=-1).view(1, 1, 4, 2)
        rebuilt = apply(tensor, layer_idx=0, target="k", bits=4, group=4)
        expected = torch.stack((_qdq(feature0, 4), _qdq(feature1, 4)), dim=-1).view(1, 1, 4, 2)
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-5))

    def test_key_tail_is_its_own_group(self):
        channel = torch.arange(40, dtype=torch.float32)
        tensor = channel.view(1, 1, 40, 1)
        rebuilt = apply(tensor, layer_idx=0, target="k", bits=8, group=32).view(40)
        expected = torch.cat((_qdq(channel[:32], 8), _qdq(channel[32:], 8)))
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-5))

    def test_single_decode_key_stays_exact(self):
        token = torch.tensor([[[[1.25, -3.5]]]])
        rebuilt = apply(token, layer_idx=0, target="k", bits=4, group=32)
        self.assertTrue(torch.allclose(rebuilt, token, atol=1e-5))

    def test_zero_group_stays_zero(self):
        tensor = torch.zeros(1, 1, 2, 8)
        self.assertTrue(torch.equal(apply(tensor, layer_idx=0, target="v", bits=8, group=4), tensor))


class QuantizeRunTests(unittest.TestCase):
    def test_rungs_climb_eight_bits_then_four(self):
        widths = tuple(rung.level for rung in rungs_for("quantize"))
        fractions = tuple(rung.fraction for rung in rungs_for("quantize"))
        self.assertEqual(widths, (8, 4))
        self.assertEqual(fractions, (0.5, 0.75))
        self.assertEqual([next_level(0, widths), next_level(8, widths), next_level(4, widths)], [8, 4, None])

    def test_sweep_scores_every_slot_at_both_widths(self):
        configurations = sweep_run()["configurations"]
        names = [item["name"] for item in configurations]
        self.assertEqual(names[0], "llama31_dense")
        self.assertEqual(len(names), 1 + 32 * 2 * 2)
        eight = next(item for item in configurations if item["name"] == "llama31_quantize_k00_p8")
        four = next(item for item in configurations if item["name"] == "llama31_quantize_v00_p4")
        self.assertEqual(eight["kv"]["pipeline"][0]["bits"], 8)
        self.assertEqual(eight["kv"]["pipeline"][0]["group"], 32)
        self.assertEqual(four["kv"]["pipeline"][0]["bits"], 4)
        self.assertNotIn("pre_rope", eight["kv"]["pipeline"][0])
        self.assertEqual(eight["tasks"], ["wikitext"])

    def test_assignment_can_mix_eight_bits_and_four_bits(self):
        kv = kv_for_assignment(quantize(), {Slot(3, "k"): 8, Slot(7, "v"): 4})
        by_bits = {step["bits"]: step for step in kv["pipeline"]}
        self.assertEqual(by_bits[8]["k_layers"], [3])
        self.assertEqual(by_bits[8]["v_layers"], [])
        self.assertEqual(by_bits[4]["k_layers"], [])
        self.assertEqual(by_bits[4]["v_layers"], [7])


if __name__ == "__main__":
    unittest.main()
