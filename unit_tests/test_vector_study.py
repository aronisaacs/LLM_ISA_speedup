"""CPU tests for the vector-compression sweep, budget run, and figures."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.compressions import vector_compress
from catalog.tasks import GSM8K_20PCT
from engine.eval_runner.execute import is_finished_result, normalize_samples
from engine.eval_runner.load_run import load_run
from engine.layer_select.apply import kv_for_slot, parse_singleton
from engine.layer_select.budgets import (
    BUDGETS,
    LLAMA31_LAYERS,
    budget_run,
    selections_for_budgets,
    write_budget_run,
    write_selections,
)
from engine.layer_select.levels import LEVELS
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots
from scripts.plot_vector_study import write_sweep_plots, write_task_tables
from scripts.vector_study import _multi_run_command

ROOT = Path(__file__).resolve().parents[1]


def _rows(n_layers):
    rows = []
    for slot in all_slots(n_layers):
        for index, level in enumerate(LEVELS):
            delta = 0.1 * (index + 1) if slot.target == "v" else 1.0 * (index + 1)
            rows.append(
                ScoreRow(
                    slot=slot,
                    level=level,
                    ppl=10.0 + delta,
                    delta=delta,
                    path=f"{slot.tag()}p{level}.json",
                )
            )
    return rows


class VectorSweepTests(unittest.TestCase):
    def test_sweep_is_wikitext_singletons_at_three_rungs(self):
        loaded = load_run(ROOT / "runs" / "vector_sweep.py")
        configurations = loaded["configurations"]
        self.assertEqual(len(configurations), 1 + LLAMA31_LAYERS * 2 * len(LEVELS))
        self.assertEqual(loaded["model_args"], "pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16")
        dense = configurations[0]
        self.assertEqual(dense["tasks"], ["wikitext"])
        self.assertNotIn("limit", dense)
        self.assertEqual(dense["kv"], {"pipeline": []})
        slot, level = parse_singleton(configurations[1]["kv"])
        self.assertEqual((slot, level), (Slot(0, "k"), 25))
        step = configurations[1]["kv"]["pipeline"][0]
        self.assertEqual(step["method"], "vector_compress")
        self.assertEqual(step["prune_pct"], 25)
        self.assertNotIn("threshold", step)
        slot, level = parse_singleton(configurations[-1]["kv"])
        self.assertEqual((slot, level), (Slot(LLAMA31_LAYERS - 1, "v"), 75))


class BudgetTests(unittest.TestCase):
    def test_six_budgets_climb_and_feed_three_tasks(self):
        payloads = selections_for_budgets(_rows(1), 1, vector_compress(prune_pct=25), dense_ppl=10.0)
        self.assertEqual([payload["tag"] for payload in payloads], ["p10", "p20", "p30", "p40", "p50", "p60"])
        self.assertEqual(tuple(payload["budget"] for payload in payloads), BUDGETS)
        compressions = [payload["compression"] for payload in payloads]
        self.assertEqual(compressions, sorted(compressions))
        self.assertGreaterEqual(compressions[0], 0.10)
        self.assertEqual(payloads[0]["assignment"], [{"layer": 0, "target": "v", "level": 25}])
        self.assertEqual(payloads[0]["kv"]["pipeline"][0]["method"], "vector_compress")

        run = budget_run(payloads)
        names = [item["name"] for item in run["configurations"]]
        self.assertEqual(len(names), 3 * (1 + len(BUDGETS)))
        self.assertEqual(names[0], "llama31_ceval_dense")
        self.assertEqual(names[2], "llama31_humaneval_instruct_dense")
        self.assertEqual(names[3], "llama31_ceval_p10")
        self.assertEqual(run["configurations"][0]["kv"], {"pipeline": []})
        self.assertEqual(run["configurations"][0]["tasks"], ["ceval-valid"])
        self.assertEqual(run["configurations"][0]["num_fewshot"], 5)
        humaneval = run["configurations"][2]
        self.assertTrue(humaneval["confirm_run_unsafe_code"])
        self.assertEqual(
            humaneval["output_path"],
            "results/vector_study/json/budgets/humaneval_instruct_dense.json",
        )
        self.assertEqual(run["configurations"][1]["samples"], GSM8K_20PCT["samples"])

    def test_write_selections_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scores = folder / "sweep"
            study = folder / "study"
            scores.mkdir()
            _dump_sweep(scores)
            payloads = write_selections(scores, study, n_layers=1)
            self.assertTrue((study / "selected_p10.json").is_file())
            index = json.loads((study / "budgets.json").read_text())
            self.assertEqual(index[0]["tag"], "dense")
            self.assertEqual(index[1]["tag"], "p10")
            run_path = write_budget_run(payloads, study / "budgets_run.json")
            loaded = load_run(run_path)
            self.assertEqual(len(loaded["configurations"]), 21)


class PlotTests(unittest.TestCase):
    def test_sweep_curves_and_task_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            scores = folder / "sweep"
            study = folder / "study"
            tasks = folder / "tasks"
            scores.mkdir()
            study.mkdir()
            tasks.mkdir()
            _dump_sweep(scores)
            (study / "budgets.json").write_text(
                json.dumps(
                    [
                        {"tag": "dense", "budget": 0.0, "compression": 0.0},
                        {"tag": "p10", "budget": 0.10, "compression": 0.10},
                    ]
                )
            )
            _dump_task(tasks / "ceval_dense.json", 0.5490)
            _dump_task(tasks / "ceval_p10.json", 0.5438)
            sweep_paths = write_sweep_plots(scores, study)
            self.assertEqual([path.name for path in sweep_paths], ["sweep_p25.svg", "sweep_p50.svg", "sweep_p75.svg"])
            curve = sweep_paths[0].read_text()
            self.assertIn("25% compression", curve)
            self.assertIn("K layers", curve)
            self.assertIn("V layers", curve)
            self.assertIn("Baseline", curve)
            self.assertIn("5% degradation band", curve)
            tables = write_task_tables(tasks, study, study)
            self.assertEqual(tables[0].name, "ceval.svg")
            table = tables[0].read_text()
            self.assertIn("Llama 3.1 8B Instruct — CEval", table)
            self.assertIn("Baseline", table)
            self.assertIn("54.90", table)
            self.assertIn("54.38", table)
            self.assertIn("0.95%", table)
            self.assertIn("10%", table)
            self.assertIn("Overall KV Compression", table)

    def test_ceval_table_uses_the_group_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            study = folder / "study"
            tasks = folder / "tasks"
            study.mkdir()
            tasks.mkdir()
            (study / "budgets.json").write_text(
                json.dumps([{"tag": "dense", "budget": 0.0, "compression": 0.0}])
            )
            (tasks / "ceval_dense.json").write_text(
                json.dumps(
                    {
                        "groups": {"ceval-valid": {"acc,none": 0.549}},
                        "results": {"ceval-valid_logic": {"acc,none": 0.10}},
                    }
                )
            )
            table = write_task_tables(tasks, study, study)[0].read_text()
            self.assertIn("54.90", table)
            self.assertNotIn("10.00", table)


class SampleFileTests(unittest.TestCase):
    def test_gsm8k_profile_is_every_fifth_test_item(self):
        payload = normalize_samples(GSM8K_20PCT["samples"])
        indices = payload["gsm8k"]
        self.assertEqual(indices, list(range(0, 1319, 5)))
        self.assertEqual(len(indices), 264)
        self.assertTrue(all(index < 1319 for index in indices))


class ResumeTests(unittest.TestCase):
    def test_finished_json_is_skippable_and_the_study_asks_for_that(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            done = folder / "done.json"
            broken = folder / "broken.json"
            done.write_text(json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.2}}}))
            broken.write_text("{")
            self.assertTrue(is_finished_result(done))
            self.assertFalse(is_finished_result(broken))
            self.assertFalse(is_finished_result(folder / "missing.json"))
        command = _multi_run_command(Path("runs/vector_sweep.py"))
        self.assertIn("--skip-existing", command)


def _dump_sweep(folder: Path) -> None:
    template = vector_compress(prune_pct=25)

    def dump(path: Path, ppl: float, kv: dict):
        payload = {
            "results": {"wikitext": {"word_perplexity,none": ppl}},
            "configs": {"wikitext": {"metadata": {"kv": kv}}},
        }
        path.write_text(json.dumps(payload))

    dump(folder / "dense.json", 10.0, {"pipeline": []})
    for slot in all_slots(1):
        for index, level in enumerate(LEVELS):
            delta = 0.1 * (index + 1) if slot.target == "v" else float(index + 1)
            dump(
                folder / f"{slot.tag()}p{level}.json",
                10.0 + delta,
                kv_for_slot(template, slot, level=level),
            )


def _dump_task(path: Path, accuracy: float) -> None:
    path.write_text(
        json.dumps({"results": {"ceval-valid": {"acc,none": accuracy}}, "configs": {}})
    )


if __name__ == "__main__":
    unittest.main()
