"""Recorded official results are evidence: check them without rewriting them."""

import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "compression_topics/spatial/figures"
METRICS = {"ceval": ("ceval-valid", "acc,none"),
           "gsm8k": ("gsm8k", "exact_match,flexible-extract"),
           "humaneval": ("humaneval_instruct", "pass@1,create_test")}


class RecordedResultsTests(unittest.TestCase):
    def test_official_summary_matches_underlying_results(self):
        folder = ROOT / "official_llama"
        path = folder / "summary.json"
        if not path.exists():
            self.skipTest("recorded official results are not available")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["status"], "complete")
        for row in summary["rows"]:
            with self.subTest(result=row["name"]):
                payload = json.loads((folder / "tasks" / f"{row['name']}.json").read_text())
                key, metric = METRICS[row["task"]]
                self.assertEqual(payload["n-samples"][key]["effective"], row["questions"])
                self.assertAlmostEqual(row["accuracy"], payload["results"][key][metric])
                if row["name"].startswith("dense_"):
                    continue
                storage = payload["storage"]["targets"]["kv"]
                self.assertAlmostEqual(row["actual_kv_saving"], storage["compression"])
                self.assertTrue(math.isclose(
                    storage["compression"],
                    1 - storage["stored_bits"] / storage["dense_bits"], abs_tol=1e-9)
                    if "stored_bits" in storage else True)


if __name__ == "__main__":
    unittest.main()
