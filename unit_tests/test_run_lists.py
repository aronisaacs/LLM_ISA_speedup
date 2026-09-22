"""Unit tests for Python/JSON run lists. Does not load a full LLM."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval_runner.device import apply_device, available_device
from eval_runner.run_list import load_run_list

ROOT = Path(__file__).resolve().parents[1]


class LoadRunListTests(unittest.TestCase):
    def test_mac_eval_runs_expands_dense_then_sparsify(self):
        config = load_run_list(ROOT / "run_lists" / "mac_eval_runs.py")
        names = [run["name"] for run in config["runs"]]
        self.assertEqual(
            names,
            [
                "llama32_1b_arc_easy_dense",
                "llama32_1b_gsm8k_dense",
                "llama32_1b_arc_easy_sparsify48",
                "llama32_1b_gsm8k_sparsify48",
            ],
        )
        self.assertEqual(config["runs"][0]["kv"], {"pipeline": []})
        self.assertEqual(config["runs"][2]["kv"]["pipeline"][0]["method"], "sparsify_nm")
        self.assertEqual(
            config["runs"][0]["output_path"],
            "mac_eval_results/llama32_1b_arc_easy_dense.json",
        )

    def test_dgx_eval_runs_groups_by_model_then_compression(self):
        config = load_run_list(ROOT / "run_lists" / "dgx_eval_runs.py")
        names = [run["name"] for run in config["runs"]]
        self.assertEqual(names[0], "llama31_ceval_dense")
        self.assertEqual(names[3], "llama31_ceval_sparsify48")
        self.assertEqual(names[-1], "codellama7b_humaneval_sparsify48")
        self.assertEqual(len(names), 14)

    def test_json_run_list_still_loads(self):
        payload = {"device": "cpu", "runs": [{"name": "one", "kv": {"pipeline": []}}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(payload))
            path = Path(handle.name)
        try:
            loaded = load_run_list(path)
            self.assertEqual(loaded["device"], "cpu")
            self.assertEqual(loaded["runs"][0]["name"], "one")
        finally:
            path.unlink()

    def test_python_run_list_requires_config(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write("RUNS = []\n")
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "must define config"):
                load_run_list(path)
        finally:
            path.unlink()

    def test_mac_eval_does_not_hardcode_mps(self):
        config = load_run_list(ROOT / "run_lists" / "mac_eval_runs.py")
        self.assertNotIn("device", config)
        self.assertNotIn("mps", config["model_args"])

    def test_available_device_is_a_known_backend(self):
        self.assertIn(available_device(), {"cuda", "mps", "cpu"})

    def test_apply_device_fills_then_respects_explicit(self):
        self.assertEqual(apply_device("pretrained=x,dtype=float16", "cuda"), "pretrained=x,dtype=float16,device=cuda")
        self.assertEqual(apply_device("pretrained=x,device=cpu", "cuda"), "pretrained=x,device=cpu")
        self.assertEqual(apply_device({"pretrained": "x"}, "mps"), {"pretrained": "x", "device": "mps"})
