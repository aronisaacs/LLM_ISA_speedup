"""Unit tests for newline progress helpers. Does not load a full LLM."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from eval_runner.progress import format_hms, kv_brief, summarize_scores


class ProgressTests(unittest.TestCase):
    def test_format_hms(self):
        self.assertEqual(format_hms(9), "9s")
        self.assertEqual(format_hms(75), "1m 15s")
        self.assertEqual(format_hms(3661), "1h 01m")

    def test_kv_brief_dense_and_sparsify(self):
        dense = SimpleNamespace(to_dict=lambda: {"pipeline": []})
        sparse = SimpleNamespace(
            to_dict=lambda: {
                "pipeline": [{"method": "sparsify_nm", "n": 8, "m": 4, "k_layers": "all", "v_layers": "all"}]
            }
        )
        checksparse = SimpleNamespace(
            to_dict=lambda: {
                "pipeline": [
                    {"method": "checksparse_l1", "tile": 8, "prune_pct": 50, "k_layers": "all", "v_layers": "all"}
                ]
            }
        )
        vector = SimpleNamespace(
            to_dict=lambda: {
                "pipeline": [{"method": "vector_compress", "threshold": 0.25, "k_layers": "all", "v_layers": "all"}]
            }
        )
        self.assertEqual(kv_brief(dense), "dense")
        self.assertEqual(kv_brief(sparse), "sparsify 8:4")
        dynprec = SimpleNamespace(
            to_dict=lambda: {
                "pipeline": [
                    {
                        "method": "dynamic_precision",
                        "tile": 8,
                        "bits": [16, 8, 4],
                        "pcts": [25, 50, 25],
                        "k_layers": "all",
                        "v_layers": "all",
                    }
                ]
            }
        )
        self.assertEqual(kv_brief(checksparse), "checksparse L1 tile=8 prune=50%")
        self.assertEqual(kv_brief(vector), "vector |x|<0.25")
        self.assertEqual(kv_brief(dynprec), "dynprec tile=8 bits=[16, 8, 4] pcts=[25, 50, 25]")

    def test_summarize_scores_picks_primary_metric(self):
        results = {
            "results": {
                "arc_easy": {"alias": "arc_easy", "acc,none": 0.5, "acc_norm,none": 0.4},
                "gsm8k": {"exact_match,flexible-extract": 0.09375},
            }
        }
        text = summarize_scores(results)
        self.assertIn("arc_easy acc,none=0.500", text)
        self.assertIn("gsm8k exact_match,flexible-extract=0.094", text)
