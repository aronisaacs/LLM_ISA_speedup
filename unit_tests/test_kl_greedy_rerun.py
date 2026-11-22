"""The KL greedy rerun: spatial window option, spatial levels, isolated greedy on KL, task runs."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from compression_topics.kl_greedy.scripts import rerun
from compression_topics.spatial.algorithms import methods, windows
from compression_topics.spatial.algorithms.formats import OFFICIAL_WINDOW, WINDOW
from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.apply import _level_from_step, kv_for_assignment
from engine.layer_select.greedy.calibrated import choose_candidates
from engine.layer_select.slots import Slot

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("wikitext", "arc_5shot", "gsm8k", "mbpp")


class SpatialWindowTests(unittest.TestCase):
    def test_default_is_32_and_written_into_the_spec(self):
        self.assertEqual(OFFICIAL_WINDOW, 32)
        step = SpatialConfig().pipeline(0.3)["pipeline"][0]
        self.assertEqual(step["window"], 32)
        parse_kv_spec({"pipeline": [step]})  # the option validates

    def test_16_writes_no_key_so_older_runs_keep_their_identity(self):
        step = SpatialConfig(window=WINDOW).pipeline(0.3)["pipeline"][0]
        self.assertNotIn("window", step)

    def test_odd_window_is_refused(self):
        with self.assertRaises(ValueError):
            SpatialConfig(window=15)

    def test_cache_hook_waits_for_a_whole_window(self):
        torch.manual_seed(0)
        x = torch.randn(1, 2, 48, 128, dtype=torch.bfloat16)
        for window, changed in ((16, 48), (32, 32)):
            methods.reset()
            cache = x.clone()
            methods.after_append(cache, target="v", layer_idx=0, start=0, end=48, saving=0.3, window=window)
            rows = (cache != x).any(-1).any(1)[0]
            self.assertEqual(int(rows.nonzero().max()) + 1 if rows.any() else 0, changed)
            self.assertTrue(torch.equal(cache[..., changed:, :], x[..., changed:, :]))

    def test_block_bits_scale_with_the_window(self):
        self.assertEqual(windows.block_bits(128, 0.25, 32), 0.75 * 32 * 128 * 16 + 16 * 2)
        out, bits, _ = windows.compress(torch.randn(1, 2, 64, 128), target="v", saving=0.25, window=32)
        self.assertEqual(bits, int(2 * 2 * windows.block_bits(128, 0.25, 32)))


class SpatialLevelTests(unittest.TestCase):
    def test_level_is_the_slot_saving_in_percent(self):
        kv = rerun.template("spatial", rerun.MODELS["llama"])
        mixed = kv_for_assignment(kv, {Slot(0, "k"): 10, Slot(3, "v"): 47})
        steps = mixed["pipeline"]
        self.assertEqual([(s["saving"], s["k_layers"], s["v_layers"]) for s in steps],
                         [(0.10, [0], []), (0.47, [], [3])])
        self.assertTrue(all(s["window"] == 32 and s["query_weights_sha256"] for s in steps))
        self.assertEqual([_level_from_step(s) for s in steps], [10, 47])
        parse_kv_spec(mixed)


class GreedyTests(unittest.TestCase):
    def test_score_key_replaces_perplexity(self):
        choices = {(0, "k"): [{"level": 1, "allocation_saving": 0.5, "kl": 1.0}],
                   (0, "v"): [{"level": 1, "allocation_saving": 0.5, "kl": 0.1}]}
        selected, total = choose_candidates(choices, 0.0, 0.25, score="kl")
        self.assertEqual(list(selected), [(0, "v")])
        self.assertEqual(total, 0.5)

    def test_net_savings(self):
        llama, qwen = rerun.MODELS["llama"], rerun.MODELS["qwen"]
        self.assertAlmostEqual(rerun.saving("vector", llama, Slot(0, "k"), 25), (32 * 16 - 128) / 2048)
        self.assertAlmostEqual(rerun.saving("vector", qwen, Slot(3, "v"), 75), (192 * 16 - 256) / 4096)
        self.assertAlmostEqual(rerun.saving("vector", llama, Slot(0, "k"), 37), 0.3047, places=4)  # reference
        self.assertAlmostEqual(rerun.saving("spatial", llama, Slot(0, "k"), 40), 0.4 - 32 / (32 * 128 * 16))

    def test_candidates_cover_kv_layers_only(self):
        found = rerun.candidates("spatial", rerun.MODELS["qwen"])
        self.assertEqual(len(found), 1 + 16 * 5)
        self.assertEqual(found["reference"], {s: 30 for s in rerun.slots(rerun.MODELS["qwen"])})
        self.assertIn("k03_10", found)
        self.assertNotIn("k00_10", found)
        self.assertEqual(len(rerun.candidates("vector", rerun.MODELS["codellama"])), 1 + 64 * 3)

    def test_allocation_reaches_each_budget_cheapest_slots_first(self):
        model = rerun.MODELS["qwen"]
        kl = {"reference": {d: 2.0 for d in DOMAINS}}
        for slot in rerun.slots(model):
            for level in rerun.METHODS["vector"].levels:
                base = 0.01 if slot.target == "v" else 0.5  # values are cheap here
                kl[f"{slot.tag()}_{level}"] = {d: base * level / 25 * (1 + slot.layer / 100) for d in DOMAINS}
        previous = 0.0
        for budget in rerun.METHODS["vector"].budgets:
            selection = rerun.allocate("vector", model, kl, budget)
            self.assertGreaterEqual(selection["compression"], budget - 1e-12)
            self.assertGreaterEqual(selection["compression"], previous)
            previous = selection["compression"]
            parse_kv_spec(selection["kv"])
        first = rerun.allocate("vector", model, kl, 0.10)
        self.assertTrue(all(a["target"] == "v" for a in first["assignment"]))


class HigherBudgetTests(unittest.TestCase):
    """The budgets added on 2026-11-22 extend the greedy path: the committed maps stay the same."""

    def test_committed_maps_are_unchanged_and_the_last_budget_is_the_maximum(self):
        for name, model in rerun.MODELS.items():
            for method, spec in rerun.METHODS.items():
                folder = ROOT / rerun.OUT / name / method
                if not (folder / "selections.json").exists():
                    self.skipTest("no committed results")
                kl = json.loads((folder / "singles.json").read_text())["kl"]
                committed = json.loads((folder / "selections.json").read_text())
                for old in committed:
                    new = rerun.allocate(method, model, kl, old["budget"])
                    self.assertEqual(new["assignment"], old["assignment"], (name, method, old["tag"]))
                    self.assertEqual(new["kv"], old["kv"])
                top = rerun.allocate(method, model, kl, spec.budgets[-1])
                self.assertTrue(all(a["level"] == spec.levels[-1] for a in top["assignment"]))
                self.assertEqual(len(top["assignment"]), 2 * len(model.kv_layers))
                self.assertEqual(len({f"p{round(b * 100):02d}" for b in spec.budgets}), len(spec.budgets))


class VectorWindowTests(unittest.TestCase):
    def test_newest_tokens_stay_exact_and_older_ones_are_pruned(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.kv_compress.install import install

        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=32)).eval()
        from catalog.compressions import vector_compress
        kv = vector_compress(k_layers=[], v_layers=[0], prune_pct=50, recent_window=32)
        uninstall = install(SimpleNamespace(model=model), parse_kv_spec(kv))
        try:
            with torch.inference_mode():
                cache = model(input_ids=torch.arange(40).unsqueeze(0) % 32, use_cache=True).past_key_values
        finally:
            uninstall()
        values = cache.layers[0].values[0, 0]
        zeros = (values == 0).sum(-1)
        self.assertTrue(bool((zeros[:8] >= 16).all()))  # aged out of the window: half the values zeroed
        self.assertTrue(bool((zeros[8:] == 0).all()))   # the newest 32 tokens: exact


class TaskRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=ROOT))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.original = rerun.out_dir
        rerun.out_dir = lambda args, name, method: self.tmp / name / method
        self.addCleanup(setattr, rerun, "out_dir", self.original)

    def _selections(self, name, method):
        model = rerun.MODELS[name]
        kl = {"reference": {d: 1.0 for d in DOMAINS}}
        kl.update({label: {d: 0.1 * len(a) * (1 + i % 7) for d in DOMAINS}
                   for i, (label, a) in enumerate(rerun.candidates(method, model).items()) if label != "reference"})
        out = self.tmp / name / method
        out.mkdir(parents=True)
        (out / "singles.json").write_text(json.dumps({"identity": {}, "kl": kl}))
        return rerun.greedy(SimpleNamespace(timing=None), name, method)

    def test_spatial_run_batches_generation_but_not_ceval(self):
        self._selections("qwen", "spatial")
        run = rerun.tasks_run(SimpleNamespace(timing=None), "qwen", "spatial")
        self.assertEqual(run["batch_size"], rerun.BATCH_SIZE)
        for c in run["configurations"]:
            self.assertEqual(c.get("batch_size", run["batch_size"]), 1 if c["tasks"] == ["ceval-valid"] else 8)
        self.assertIn("Qwen3.5-9B", run["model_args"])
        configs = run["configurations"]
        per_task = 1 + len(rerun.METHODS["spatial"].budgets)
        self.assertEqual(len(configs), 3 * per_task)
        self.assertEqual([c["tasks"][0] for c in configs[:per_task]], ["gsm8k"] * per_task)
        self.assertEqual(configs[0]["name"], "qwen_spatial_gsm8k_dense")
        self.assertEqual(configs[-1]["tasks"], ["ceval-valid"])

    def test_vector_tasks_keep_32_tokens_exact_but_calibration_does_not(self):
        selections = self._selections("llama", "vector")
        for selection in selections:
            tasks = selection["kv"]["pipeline"]
            scored = selection["calibration_kv"]["pipeline"]
            self.assertTrue(all(step["recent_window"] == 32 for step in tasks))
            self.assertTrue(all("recent_window" not in step for step in scored))
            self.assertEqual([{k: v for k, v in step.items() if k != "recent_window"} for step in tasks], scored)
            parse_kv_spec(selection["kv"])
        self.assertTrue(all("recent_window" not in s for s in rerun.template("vector", rerun.MODELS["llama"])["pipeline"]))
        run = rerun.tasks_run(SimpleNamespace(timing=None), "llama", "vector")
        compressed = [c for c in run["configurations"] if not c["name"].endswith("_dense")]
        self.assertTrue(all(step["recent_window"] == 32 for c in compressed for step in c["kv"]["pipeline"]))

    def test_spatial_maps_have_no_recent_window(self):
        for selection in self._selections("qwen", "spatial"):
            self.assertEqual(selection["kv"], selection["calibration_kv"])

    def test_codellama_runs_humaneval_with_the_indent_prompt(self):
        self._selections("codellama", "vector")
        run = rerun.tasks_run(SimpleNamespace(timing=None), "codellama", "vector")
        self.assertEqual({c["tasks"][0] for c in run["configurations"]}, {"humaneval_codellama"})
        self.assertEqual(len(run["configurations"]), 1 + len(rerun.METHODS["vector"].budgets))
        self.assertEqual(run["batch_size"], rerun.BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
