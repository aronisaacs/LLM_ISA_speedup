"""A finished JSON is reused by model, task, and kv, not by output path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.eval_runner.cache import reuse_cached_result
from engine.kv_compress.spec import parse_kv_spec

_MODEL = "pretrained=test/cache-model,dtype=bfloat16"


def _payload(kv, task="wikitext", fewshot=0):
    return {
        "results": {task: {"acc,none": 0.5}},
        "group_subtasks": {task: []},
        "n-shot": {task: fewshot},
        "configs": {task: {"metadata": {"kv": kv}}},
        "config": {
            "model_args": {"pretrained": "test/cache-model", "dtype": "bfloat16", "kv": kv},
            "limit": None,
            "gen_kwargs": None,
        },
    }


class CacheReuseTests(unittest.TestCase):
    def test_same_simulation_in_another_folder_is_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old" / "dense.json"
            source.parent.mkdir()
            kv = {"pipeline": []}
            source.write_text(json.dumps(_payload(kv)))
            dest = root / "new" / "again.json"
            base = {"model_args": _MODEL, "tasks": ["wikitext"], "num_fewshot": 0}
            kind = reuse_cached_result(base, {}, parse_kv_spec(kv), dest, skip_unkeyed=False)
            self.assertEqual(kind, "reused")
            self.assertEqual(json.loads(dest.read_text())["results"]["wikitext"]["acc,none"], 0.5)

    def test_a_different_pipeline_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dense.json"
            source.write_text(json.dumps(_payload({"pipeline": []})))
            dest = root / "sparse.json"
            kv = {"pipeline": [{"method": "vector_compress", "k_layers": "all", "v_layers": [], "prune_pct": 25}]}
            base = {"model_args": _MODEL, "tasks": ["wikitext"], "num_fewshot": 0}
            kind = reuse_cached_result(base, {}, parse_kv_spec(kv), dest, skip_unkeyed=False)
            self.assertIsNone(kind)
            self.assertFalse(dest.exists())

    def test_destination_already_holding_the_simulation_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "done.json"
            kv = {"pipeline": []}
            dest.write_text(json.dumps(_payload(kv)))
            base = {"model_args": _MODEL, "tasks": ["wikitext"], "num_fewshot": 0}
            kind = reuse_cached_result(base, {}, parse_kv_spec(kv), dest, skip_unkeyed=False)
            self.assertEqual(kind, "skip")


if __name__ == "__main__":
    unittest.main()
