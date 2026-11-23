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
        self.assertEqual(len(rerun.candidates("vector", rerun.MODELS["codellama"])), 1 + 64 * 6)

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

    def test_committed_maps_are_unchanged(self):
        for name, model in rerun.MODELS.items():
            for method, spec in rerun.METHODS.items():
                folder = ROOT / rerun.OUT / name / method
                if not (folder / "selections.json").exists():
                    continue  # not run yet
                kl = json.loads((folder / "singles.json").read_text())["kl"]
                if any(label not in kl for label in rerun.candidates(method, model)):
                    continue  # levels added since (vector eighths, 2026-11-23): every map changes, not extends
                committed = json.loads((folder / "selections.json").read_text())
                for old in committed:
                    if old["calibration"]["allocator"] == "uniform":  # p50: a fixed map, not a budget
                        continue
                    new = rerun.allocate(method, model, kl, old["budget"])
                    self.assertEqual(new["assignment"], old["assignment"], (name, method, old["tag"]))
                    self.assertEqual(new["kv"], old["kv"])
                self.assertEqual(len({f"p{round(b * 100):02d}" for b in spec.budgets}), len(spec.budgets))


class TurboquantTests(unittest.TestCase):
    """TurboQuant: levels 4 / 3 / 2 bits, every map starts from uniform 4 bits (Aron, 2026-11-22)."""

    def test_net_savings_count_one_norm_per_128_dims(self):
        for name in ("llama", "qwen"):
            model = rerun.MODELS[name]
            slot = rerun.slots(model)[0]
            for bits in (4, 3, 2):
                self.assertAlmostEqual(rerun.saving("turboquant", model, slot, bits), 1 - (bits + 0.125) / 16)

    def test_singles_change_one_slot_of_the_uniform_4_bit_map(self):
        model = rerun.MODELS["qwen"]
        found = rerun.candidates("turboquant", model)
        self.assertEqual(len(found), 2 + 16 * 2)
        everything = set(rerun.slots(model))
        self.assertEqual(found["base"], {s: 4 for s in everything})
        self.assertEqual(found["reference"], {s: 3 for s in everything})
        single = found["v07_2"]
        self.assertEqual(single[Slot(7, "v")], 2)
        self.assertEqual({lvl for s, lvl in single.items() if s != Slot(7, "v")}, {4})
        self.assertEqual(len(rerun.candidates("turboquant", rerun.MODELS["llama"])), 2 + 64 * 2)

    def _kl(self, model):
        kl = {"reference": {d: 2.0 for d in DOMAINS}, "base": {d: 0.5 for d in DOMAINS}}
        for slot in rerun.slots(model):
            for bits in (3, 2):
                cost = (0.01 if slot.target == "v" else 0.2) * (4 - bits) * (1 + slot.layer / 100)
                kl[f"{slot.tag()}_{bits}"] = {d: 0.5 + cost for d in DOMAINS}
        return kl

    def test_allocation_starts_at_4_bits_and_reaches_each_budget(self):
        for name in ("llama", "qwen"):
            model = rerun.MODELS[name]
            kl = self._kl(model)
            previous = 0.0
            for budget in rerun.METHODS["turboquant"].budgets:
                selection = rerun.allocate("turboquant", model, kl, budget)
                self.assertGreaterEqual(selection["compression"], budget - 1e-12)
                self.assertGreater(selection["compression"], previous)
                previous = selection["compression"]
                self.assertEqual(len(selection["assignment"]), len(rerun.slots(model)))
                self.assertLessEqual({a["level"] for a in selection["assignment"]}, {4, 3, 2})
                parse_kv_spec(selection["kv"])
                parse_kv_spec(selection["calibration_kv"])
            first = rerun.allocate("turboquant", model, kl, 0.78)
            lowered = [a for a in first["assignment"] if a["level"] != 4]
            self.assertTrue(lowered and all(a["target"] == "v" for a in lowered))  # values are cheap here

    def test_windows_and_qwen_halves(self):
        for name in ("llama", "qwen"):
            model = rerun.MODELS[name]
            selection = rerun.allocate("turboquant", model, self._kl(model), 0.81)
            for step in selection["kv"]["pipeline"]:
                self.assertEqual(step["recent_window"], 32)
                self.assertEqual(step["signs"], "rope_pairs")
            for step in selection["calibration_kv"]["pipeline"]:
                self.assertNotIn("recent_window", step)
            for step in selection["kv"]["pipeline"] + selection["calibration_kv"]["pipeline"]:
                if name == "qwen":
                    self.assertEqual((step["part_size"], step["rope_dim"]), (128, 64))
                else:
                    self.assertNotIn("part_size", step)
                    self.assertNotIn("rope_dim", step)

    def test_uniform_maps_are_b4_and_b2(self):
        model = rerun.MODELS["codellama"]
        b4, b2 = (rerun.uniform("turboquant", model, bits) for bits in (4, 2))
        self.assertEqual((b4["tag"], b2["tag"]), ("b4", "b2"))
        self.assertAlmostEqual(b4["compression"], 1 - 4.125 / 16)
        self.assertAlmostEqual(b2["compression"], 1 - 2.125 / 16)
        self.assertEqual([s["bits"] for s in b2["kv"]["pipeline"]], [2])
        parse_kv_spec(b2["kv"])

    def test_vector_and_spatial_allocation_is_unchanged_by_the_base_option(self):
        # No base level: start 0, same budget, same assignment as before the change.
        self.assertEqual(rerun.METHODS["vector"].base, 0)
        self.assertEqual(rerun.METHODS["spatial"].base, 0)


class FakeProcess:
    """A task run that finishes after ``ticks`` polls and then has its score in ``results``."""

    def __init__(self, item, results, scores, ticks):
        self.item, self.results, self.scores, self.ticks, self.stopped = item, results, scores, ticks, False

    def poll(self):
        self.ticks -= 1
        if self.ticks > 0:
            return None
        self.results[self.item.key] = self.scores[self.item.tag]
        return 0

    def terminate(self):
        self.stopped = True

    def wait(self):
        return 0


def _chain(name, tags, uniform=()):
    return [rerun.Item({"name": f"{name}_{t}", "tasks": ["gsm8k"]}, t, 10.0, f"{name}/{t}",
                       t != "dense" and t not in uniform) for t in tags]


class CutoffTests(unittest.TestCase):
    """Each model-task stops after its first budget more than 5 points below dense (Aron, 2026-11-23)."""

    def test_wanted_keeps_the_first_budget_past_the_cutoff_and_the_uniform_maps(self):
        items = _chain("a", ["dense", "p10", "p20", "p30", "p40", "b4"], uniform=("b4",))
        scores = {"a/dense": 80.0, "a/p10": 79.0, "a/p20": 74.9, "a/p30": 70.0}
        kept = rerun.wanted(items, lambda item: scores.get(item.key))
        self.assertEqual([i.tag for i in kept], ["dense", "p10", "p20", "b4"])
        self.assertEqual(len(rerun.wanted(items, lambda item: None)), len(items))  # nothing scored: all wanted

    def test_dispatch_stops_each_chain_at_its_cutoff(self):
        found = {("m", "vector", "a"): _chain("a", ["dense", "p10", "p20", "p30", "p40", "p50"]),
                 ("m", "vector", "b"): _chain("b", ["dense", "p10", "p20"])}
        scores = {"dense": 80.0, "p10": 79.0, "p20": 78.0, "p30": 60.0, "p40": 50.0, "p50": 40.0}
        results, started = {}, []

        def start(full, device):
            item = next(i for items in found.values() for i in items if i.full is full)
            started.append(item.key)
            proc = FakeProcess(item, results, scores, ticks=1 + len(started) % 3)
            return proc

        failed = rerun.dispatch(found, ["0", "1"], start, lambda item: results.get(item.key), wait=0, say=lambda t: None)
        self.assertEqual(failed, [])
        self.assertEqual(len(started), len(set(started)))
        for key in ("a/dense", "a/p10", "a/p20", "a/p30", "b/dense", "b/p10", "b/p20"):
            self.assertIn(key, results)
        self.assertNotIn("a/p50", started)  # two past the cutoff: never started

    def test_chains_order_dense_budgets_then_uniform_and_are_self_contained(self):
        done = [(n, m) for n in rerun.MODELS for m in rerun.METHODS  # pairs with committed maps
                if (ROOT / rerun.OUT / n / m / "selections.json").exists()]
        for (name, method, task), items in rerun.chains(SimpleNamespace(timing=None), done).items():
            self.assertEqual(items[0].tag, "dense")
            budgets = [i.tag for i in items if i.budget]
            self.assertEqual(budgets, sorted(budgets, key=lambda t: int(t[1:])), (name, method, task))
            self.assertEqual(len({i.key for i in items}), len(items))
            for item in items:
                self.assertIn("pretrained=", item.full["model_args"])
                self.assertIn("batch_size", item.full)


class MergeAllTests(unittest.TestCase):
    def test_vector_eighths_zero_exact_value_counts(self):
        # 37.5 / 62.5 / 87.5 % (Aron, 2026-11-23): eighths of the vector, at head 128 and 256.
        from compression_topics.vector.algorithms.vector_compress import _apply

        for head in (128, 256):
            x = torch.arange(1, head + 1, dtype=torch.float32).repeat(3, 1)
            for level in rerun.METHODS["vector"].levels:
                zeros = int((_apply(x, threshold=0.0, prune_pct=level) == 0).sum(-1)[0])
                self.assertEqual(zeros, head * level / 100, (head, level))
        model = rerun.MODELS["llama"]
        for level, net in ((37.5, 0.3125), (62.5, 0.5625), (87.5, 0.8125), (25, 0.1875), (37, 0.3047)):
            self.assertAlmostEqual(rerun.saving("vector", model, Slot(0, "k"), level), net, places=4)

    def test_fractional_levels_survive_the_kv_spec(self):
        template = rerun.template("vector", rerun.MODELS["llama"], tasks=True)
        kv = kv_for_assignment(template, {Slot(0, "k"): 37.5, Slot(1, "v"): 75, Slot(2, "k"): 87.5})
        self.assertEqual([step["prune_pct"] for step in kv["pipeline"]], [37.5, 75, 87.5])
        self.assertEqual([_level_from_step(step) for step in kv["pipeline"]], [37.5, 75, 87.5])
        parse_kv_spec(kv)

    def test_singles_accept_added_levels_only(self):
        old = {"model": "m", "levels": [25, 50, 75], "reference_level": 37}
        self.assertTrue(rerun.same_settings(old, {**old, "levels": [25, 37.5, 50, 62.5, 75, 87.5]}))
        self.assertFalse(rerun.same_settings(old, {**old, "levels": [25, 50]}))
        self.assertFalse(rerun.same_settings(old, {**old, "levels": [25, 37.5, 50, 62.5, 75], "reference_level": 30}))

    def test_p50_merges_every_pair_with_no_residual(self):
        for name, model in rerun.MODELS.items():
            selection = rerun.uniform("spatial", model, 50)
            self.assertEqual(selection["tag"], "p50")
            self.assertEqual({a["level"] for a in selection["assignment"]}, {50})
            self.assertEqual(len(selection["assignment"]), 2 * len(model.kv_layers))
            self.assertAlmostEqual(selection["compression"], 0.4917 if model.head_dim == 128 else 0.4958, places=4)
            self.assertEqual([s["saving"] for s in selection["kv"]["pipeline"]], [0.5])
            parse_kv_spec(selection["kv"])
        torch.manual_seed(0)
        for dim in (128, 256):
            _, _, counts = windows.compress(torch.randn(1, 2, 64, dim), target="k", saving=0.5, window=32)
            self.assertEqual(counts, {"format_dense_pair": 0, "format_g2_r0": 2 * 64 // 2})


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
        per_task = 2 + len(rerun.METHODS["spatial"].budgets)  # dense, budgets, p50
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

    def test_turboquant_run_has_dense_budgets_and_both_uniform_maps(self):
        self._selections("codellama", "turboquant")
        run = rerun.tasks_run(SimpleNamespace(timing=None), "codellama", "turboquant")
        tags = [c["output_path"].rsplit("_", 1)[1].removesuffix(".json") for c in run["configurations"]]
        self.assertEqual(sorted(tags), sorted(["dense", "p78", "p81", "p84", "b4", "b2"]))

    def test_codellama_runs_humaneval_with_the_indent_prompt(self):
        self._selections("codellama", "vector")
        run = rerun.tasks_run(SimpleNamespace(timing=None), "codellama", "vector")
        self.assertEqual({c["tasks"][0] for c in run["configurations"]}, {"humaneval_codellama"})
        self.assertEqual(len(run["configurations"]), 1 + len(rerun.METHODS["vector"].budgets))  # dense, budgets
        self.assertEqual(run["batch_size"], rerun.BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
