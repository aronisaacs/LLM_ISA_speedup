"""The committed results index is the record of every finished simulation."""

from __future__ import annotations

import unittest

from engine.eval_runner.index import simulations
from engine.layer_select.scores import load_sweep_scores

_PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"


class ShippedIndexTests(unittest.TestCase):
    def test_every_row_is_a_simulation_with_scores(self):
        rows = simulations()
        self.assertGreaterEqual(len(rows), 200)
        for row in rows:
            self.assertIn("pretrained", row["identity"])
            self.assertIn("kv", row["identity"])
            self.assertIn("tasks", row["identity"])
            self.assertTrue(row["scores"])
            self.assertNotIn("path", row)
            self.assertIn("effective", row["samples"])

    def test_vector_sweep_reads_from_the_index(self):
        dense, rows = load_sweep_scores(method="vector_compress", pretrained=_PRETRAINED)
        self.assertAlmostEqual(dense, 8.825, places=2)
        self.assertEqual(len(rows), 32 * 2 * 3)
        self.assertEqual(rows[0].kv["pipeline"][0]["method"], "vector_compress")

    def test_budget_rows_store_the_target_and_the_realized_compression(self):
        budgeted = [row for row in simulations() if "budget" in row]
        self.assertGreaterEqual(len(budgeted), 1)
        for row in budgeted:
            self.assertIn("compression", row)
            self.assertGreaterEqual(row["compression"], 0.0)


if __name__ == "__main__":
    unittest.main()
