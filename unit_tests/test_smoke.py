"""End-to-end smoke: multi_run.py on a tiny model, dense, 8 arc_easy questions.

Checks that the code still runs, not a research score. Every run goes to a
temporary --results-root (or --dry-run), so the repo's results.json is never
written. Skips when lm_eval is missing. The first run downloads the model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from engine.eval_runner.index import index_path, simulations

_ROOT = Path(__file__).resolve().parents[1]
_MULTI_RUN = _ROOT / "engine" / "multi_run.py"
_PRETRAINED = "HuggingFaceTB/SmolLM2-135M-Instruct"
_RUN = {
    "model": "hf",
    "batch_size": 1,
    "apply_chat_template": True,
    "configurations": [
        {
            "name": "smollm2_135m_arc_easy_dense_smoke",
            "model_args": f"pretrained={_PRETRAINED},dtype=float32",
            "kv": {"pipeline": []},
            "tasks": ["arc_easy"],
            "num_fewshot": 0,
            "limit": 8,
        }
    ],
}


@unittest.skipIf(importlib.util.find_spec("lm_eval") is None, "lm_eval is not installed")
class SmokeTests(unittest.TestCase):
    def setUp(self):
        self._production = _digest(index_path())
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.run_file = self.tmp / "smoke_run.json"
        self.run_file.write_text(json.dumps(_RUN))

    def tearDown(self):
        self._tmp.cleanup()
        self.assertEqual(_digest(index_path()), self._production, "smoke run touched the repo results.json")

    def test_results_root_records_one_row_there(self):
        root = self.tmp / "results_root"
        _multi_run(self, "--run", str(self.run_file), "--results-root", str(root))
        rows = simulations(root)
        self.assertEqual(len(rows), 1)
        identity = rows[0]["identity"]
        self.assertEqual(identity["pretrained"], _PRETRAINED)
        self.assertEqual(identity["limit"], 8)
        self.assertIn("arc_easy", rows[0]["scores"])

    def test_dry_run_records_nothing(self):
        root = self.tmp / "dry_root"
        _multi_run(self, "--run", str(self.run_file), "--results-root", str(root), "--dry-run")
        self.assertFalse(index_path(root).exists())


def _multi_run(test: unittest.TestCase, *args: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(_MULTI_RUN), *args],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    test.assertEqual(completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-4000:])


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


if __name__ == "__main__":
    unittest.main()
