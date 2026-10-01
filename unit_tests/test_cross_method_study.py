"""CPU tests for the interleaved checksparse + sparsify driver."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.layer_select.apply import kv_for_slot, parse_singleton
from engine.layer_select.budgets import BUDGETS, LLAMA31_LAYERS, TASKS
from engine.layer_select.levels import LEVELS
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots

import compression_topics.cross_method_study as study


def _scored(method_kv):
    rows = []
    levels = [rung.level for rung in rungs_for(method_kv["pipeline"][0]["method"])]
    for slot in all_slots(1):
        for index, level in enumerate(levels):
            delta = 0.1 * (index + 1) if slot.target == "v" else 1.0 * (index + 1)
            rows.append(
                ScoreRow(
                    slot=slot,
                    level=level,
                    ppl=10.0 + delta,
                    delta=delta,
                    path="",
                    kv=kv_for_slot(method_kv, slot, level=level),
                )
            )
    return 10.0, rows


class CrossSweepTests(unittest.TestCase):
    def test_both_methods_share_one_dense(self):
        configurations = study.sweep_run()["configurations"]
        checksparse = LLAMA31_LAYERS * 2 * len(LEVELS)
        sparsify = LLAMA31_LAYERS * 2
        self.assertEqual(len(configurations), 1 + checksparse + sparsify)
        self.assertEqual(configurations[0]["kv"], {"pipeline": []})
        methods = [c["kv"]["pipeline"][0]["method"] for c in configurations[1:]]
        self.assertEqual(methods[:checksparse], ["checksparse_l1"] * checksparse)
        self.assertEqual(methods[checksparse:], ["sparsify_nm"] * sparsify)
        first_sparsify = configurations[1 + checksparse]
        self.assertEqual(parse_singleton(first_sparsify["kv"]), (Slot(0, "k"), 50))
        self.assertEqual(set(first_sparsify["kv"]["pipeline"][0]), {"method", "k_layers", "v_layers"})
        self.assertEqual(first_sparsify["name"], "llama31_sparsify_k00_p50")
        self.assertTrue(first_sparsify["output_path"].startswith("compression_topics/sparsify/figures/"))


class CrossBudgetTests(unittest.TestCase):
    def test_one_run_holds_both_methods(self):
        templates = {method: kv for method, _tag, kv, _figures in study.METHODS}

        def fake_scores(method, pretrained):
            return _scored(templates[method])

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            methods = tuple(
                (method, tag, kv, folder / tag) for method, tag, kv, _figures in study.METHODS
            )
            with mock.patch.object(study, "METHODS", methods), mock.patch.object(
                study, "load_sweep_scores", side_effect=fake_scores
            ), mock.patch.object(study, "_rel", side_effect=str):
                path = study.write_budgets_run(folder / "budgets_run.json")
            run = json.loads(path.read_text())
            for _method, tag, _kv, figures in methods:
                self.assertTrue((figures / "selected_p10.json").is_file())

        configurations = run["configurations"]
        checksparse = len(TASKS) * len(BUDGETS)
        sparsify = len(TASKS) * len([b for b in BUDGETS if b <= 0.5])
        self.assertEqual(len(configurations), len(TASKS) + checksparse + sparsify)
        compressed = [c["kv"]["pipeline"][0]["method"] for c in configurations if c["kv"]["pipeline"]]
        self.assertEqual(compressed.count("checksparse_l1"), checksparse)
        self.assertEqual(compressed.count("sparsify_nm"), sparsify)
        names = [c["name"] for c in configurations if c["output_path"].startswith(str(folder / "sparsify"))]
        self.assertNotIn("llama31_ceval_p60", names)
        full = [c for c in configurations if c["output_path"] == str(folder / "sparsify" / "ceval_p50.json")][0]
        step = full["kv"]["pipeline"][0]
        self.assertEqual(step, {"method": "sparsify_nm", "k_layers": [0], "v_layers": [0]})


class RungTests(unittest.TestCase):
    def test_sparsify_is_on_or_off_at_half(self):
        self.assertEqual([(r.level, r.fraction) for r in rungs_for("sparsify_nm")], [(50, 0.5)])
        self.assertEqual(study.reachable_budgets("sparsify_nm"), (0.10, 0.20, 0.30, 0.40, 0.50))
        self.assertEqual(study.reachable_budgets("checksparse_l1"), BUDGETS)


class CommandTests(unittest.TestCase):
    def test_flags_are_forwarded(self):
        plain = study._multi_run_command(Path("run.json"))
        self.assertIn("--skip-existing", plain)
        self.assertNotIn("--dry-run", plain)
        self.assertNotIn("--results-root", plain)
        full = study._multi_run_command(Path("run.json"), Path("/tmp/out"), True)
        self.assertEqual(full[-3:], ["--results-root", "/tmp/out", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
