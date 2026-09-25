"""Unit tests for loading runs and expanding configurations. Does not load a full LLM."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.compressions import CHECKSPARSE_L1_50, DENSE, DYNAMIC_PRECISION_1684, SPARSIFY_48, vector_compress
from engine.eval_runner.device import apply_device, available_device
from engine.eval_runner.load_run import load_run
from engine.kv_compress.spec import parse_kv_spec


class LoadRunTests(unittest.TestCase):
    def test_json_run_still_loads(self):
        payload = {
            "device": "cpu",
            "configurations": [{"name": "one", "kv": {"pipeline": []}}],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(payload))
            path = Path(handle.name)
        try:
            loaded = load_run(path)
            self.assertEqual(loaded["device"], "cpu")
            self.assertEqual(loaded["configurations"][0]["name"], "one")
        finally:
            path.unlink()

    def test_python_run_requires_run_function(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write("CONFIGURATIONS = []\n")
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "must define run"):
                load_run(path)
        finally:
            path.unlink()

    def test_available_device_is_a_known_backend(self):
        self.assertIn(available_device(), {"cuda", "mps", "cpu"})

    def test_apply_device_fills_then_respects_explicit(self):
        self.assertEqual(apply_device("pretrained=x,dtype=float16", "cuda"), "pretrained=x,dtype=float16,device=cuda")
        self.assertEqual(apply_device("pretrained=x,device=cpu", "cuda"), "pretrained=x,device=cpu")
        self.assertEqual(apply_device({"pretrained": "x"}, "mps"), {"pretrained": "x", "device": "mps"})

    def test_catalog_compressions_parse(self):
        self.assertTrue(parse_kv_spec(DENSE).is_identity())
        sparse = parse_kv_spec(SPARSIFY_48)
        self.assertEqual(sparse.pipeline[0].method, "sparsify_nm")
        check = parse_kv_spec(CHECKSPARSE_L1_50)
        self.assertEqual(check.pipeline[0].method, "checksparse_l1")
        self.assertEqual(check.pipeline[0].kwargs["prune_pct"], 50)
        vector = parse_kv_spec(vector_compress(threshold=0.1))
        self.assertEqual(vector.pipeline[0].method, "vector_compress")
        self.assertEqual(vector.pipeline[0].kwargs["threshold"], 0.1)
        dyn = parse_kv_spec(DYNAMIC_PRECISION_1684)
        self.assertEqual(dyn.pipeline[0].method, "dynamic_precision")
        self.assertEqual(dyn.pipeline[0].kwargs["bits"], [16, 8, 4])
        self.assertEqual(dyn.pipeline[0].kwargs["pcts"], [25, 50, 25])
