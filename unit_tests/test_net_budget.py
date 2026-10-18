"""CPU tests for bitmask net savings and the net-budget greedy."""

from __future__ import annotations

import unittest

from catalog.compressions import vector_compress_pair
from compression_topics.vector.bitmask import bitmask_bits, net_saving, net_saving_for
from engine.layer_select.budgets import selections_for_budgets
from engine.layer_select.levels import LEVELS
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import all_slots


def _rows(n_layers):
    rows = []
    for slot in all_slots(n_layers):
        for index, level in enumerate(LEVELS):
            delta = 0.1 * (index + 1) if slot.target == "v" else 1.0 * (index + 1)
            rows.append(ScoreRow(slot=slot, level=level, ppl=10.0 + delta, delta=delta, path=f"{slot.tag()}p{level}"))
    return rows


class BitmaskTests(unittest.TestCase):
    def test_mask_is_one_bit_per_value_or_per_key_pair(self):
        self.assertEqual(bitmask_bits("vector_compress", "k"), 128)
        self.assertEqual(bitmask_bits("vector_compress_pair", "k"), 64)
        self.assertEqual(bitmask_bits("vector_compress_pair", "v"), 128)

    def test_net_saving_per_rung(self):
        self.assertEqual([net_saving("vector_compress", "k", level) for level in LEVELS], [0.1875, 0.4375, 0.6875])
        self.assertEqual([net_saving("vector_compress_pair", "k", level) for level in LEVELS],
                         [0.21875, 0.46875, 0.71875])
        self.assertEqual(net_saving("vector_compress_pair", "v", 50), 0.4375)
        self.assertEqual(net_saving("vector_compress", "v", 0), 0.0)


class HeadDimTests(unittest.TestCase):
    def test_net_saving_at_256_values(self):
        self.assertEqual([net_saving("vector_compress", "k", level, 256) for level in LEVELS], [0.1875, 0.4375, 0.6875])
        self.assertEqual(net_saving("vector_compress_pair", "k", 50, 256), 0.46875)


class NetBudgetTests(unittest.TestCase):
    def test_greedy_uses_only_kv_layers(self):
        layers = (1, 3)
        payload, = selections_for_budgets(_rows(4), 4, vector_compress_pair(prune_pct=25), 10.0, budgets=(0.3,),
                                          saving=net_saving_for("vector_compress"), layers=layers)
        self.assertTrue(payload["assignment"])
        self.assertTrue(all(row["layer"] in layers for row in payload["assignment"]))
        net = sum(net_saving("vector_compress", row["target"], row["level"]) for row in payload["assignment"]) / 4
        self.assertAlmostEqual(payload["compression"], net)
        self.assertGreaterEqual(payload["compression"], 0.3)

    def test_greedy_reaches_net_budget_and_keeps_zeroed_fraction(self):
        rows, n_layers = _rows(4), 4
        method = "vector_compress_pair"
        saving = net_saving_for(method)
        for payload in selections_for_budgets(rows, n_layers, vector_compress_pair(prune_pct=25), 10.0,
                                              budgets=(0.2, 0.45), saving=saving):
            self.assertGreaterEqual(payload["compression"], payload["budget"] - 1e-12)
            net = sum(net_saving(method, row["target"], row["level"]) for row in payload["assignment"]) / (2 * n_layers)
            zeroed = sum(row["level"] / 100 for row in payload["assignment"]) / (2 * n_layers)
            self.assertAlmostEqual(payload["compression"], net)
            self.assertAlmostEqual(payload["zeroed"], zeroed)
            self.assertGreater(payload["zeroed"], payload["compression"])

    def test_default_greedy_is_unchanged(self):
        payload, = selections_for_budgets(_rows(4), 4, vector_compress_pair(prune_pct=25), 10.0, budgets=(0.3,))
        self.assertNotIn("zeroed", payload)
        zeroed = sum(row["level"] / 100 for row in payload["assignment"]) / 8
        self.assertAlmostEqual(payload["compression"], zeroed)


if __name__ == "__main__":
    unittest.main()
