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
        self.assertEqual(kv_brief(dense), "dense")
        self.assertEqual(kv_brief(sparse), "sparsify 8:4")

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
