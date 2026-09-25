"""The result index is one row per simulation, not one row per run folder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.eval_runner.index import find_result, rebuild, record_result, record_simulation

_PAYLOAD = {
    "results": {"wikitext": {"word_perplexity,none": 8.5, "word_perplexity_stderr,none": 0.1, "alias": "wikitext"}},
    "group_subtasks": {"wikitext": []},
    "n-shot": {"wikitext": 0},
    "configs": {"wikitext": {"metadata": {"kv": {"pipeline": []}}}},
    "config": {
        "model_args": {"pretrained": "test/index-model", "dtype": "bfloat16", "kv": {"pipeline": []}},
        "limit": None,
        "gen_kwargs": None,
    },
}


class ResultIndexTests(unittest.TestCase):
    def test_rebuild_keeps_one_row_and_the_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "study_a" / "dense.json"
            second = root / "study_b" / "again.json"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text(json.dumps(_PAYLOAD))
            second.write_text(json.dumps(_PAYLOAD))
            rows = rebuild(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scores"]["wikitext"]["word_perplexity,none"], 8.5)
            self.assertNotIn("alias", rows[0]["scores"]["wikitext"])
            self.assertNotIn("path", rows[0])
            found = find_result(rows[0]["identity"], root)
            self.assertEqual(found["scores"]["wikitext"]["word_perplexity,none"], 8.5)

    def test_record_does_not_replace_the_first_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text(json.dumps(_PAYLOAD))
            second.write_text(json.dumps(_PAYLOAD))
            row = record_result(first, root)
            record_result(second, root)
            self.assertEqual(len(json.loads((root / "results.json").read_text())["simulations"]), 1)
            self.assertEqual(find_result(row["identity"], root)["scores"]["wikitext"]["word_perplexity,none"], 8.5)

    def test_record_simulation_keeps_samples_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"pretrained": "m", "dtype": "bfloat16", "tasks": ["gsm8k"], "kv": {"pipeline": []}}
            row = record_simulation(
                identity,
                {"gsm8k": {"exact_match,flexible-extract": 0.5}},
                samples={"original": 32, "effective": 32},
                budget=0.1,
                compression=0.12,
                root=root,
            )
            again = record_simulation(identity, {"gsm8k": {"exact_match,flexible-extract": 0.9}}, root=root)
            self.assertEqual(again["samples"]["effective"], 32)
            self.assertEqual(row["budget"], 0.1)
            self.assertEqual(len(json.loads((root / "results.json").read_text())["simulations"]), 1)


if __name__ == "__main__":
    unittest.main()
