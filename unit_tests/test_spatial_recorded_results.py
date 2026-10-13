"""Recorded presentation results are evidence: check them without rewriting them."""

import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "compression_topics/spatial/figures"


class RecordedResultsTests(unittest.TestCase):
    def test_five_rung_summaries_and_bytes_match_underlying_results(self):
        folders = [
            "clean_online_ladder_pairs",
            "clean_online_ladder_hierarchy",
            "clean_online_ladder_query_hierarchy",
            "online_fixed_rungs_no_residual",
            "online_fixed_rungs_fixed_residual",
        ]
        for folder in folders:
            path = ROOT / folder / "summary.json"
            if not path.exists():
                self.skipTest("recorded presentation data is not available")
            summary = json.loads(path.read_text())
            self.assertEqual(summary["status"], "complete")
            for row in summary["rows"]:
                with self.subTest(folder=folder, result=row["name"]):
                    payload = json.loads(
                        (ROOT / folder / "tasks" / f"{row['name']}.json").read_text()
                    )
                    self.assertEqual(
                        payload["n-samples"]["ceval-valid"]["effective"], 1346
                    )
                    self.assertAlmostEqual(
                        row["accuracy"], payload["results"]["ceval-valid"]["acc,none"]
                    )
                    if row["name"] == "dense_ceval":
                        continue
                    stats = payload["gate_stats"]
                    saved = sum(
                        r.get("dense_bits", 0) - r.get("stored_bits", 0)
                        for r in stats.values()
                    )
                    storage = payload["storage"]["targets"]["kv"]
                    self.assertTrue(
                        math.isclose(
                            storage["compression"],
                            saved / storage["dense_bits"],
                            abs_tol=1e-9,
                        )
                    )
                    self.assertAlmostEqual(
                        row["actual_prefill_kv_saving"], storage["compression"]
                    )


if __name__ == "__main__":
    unittest.main()
