"""CPU tests for the joint-KL greedy (fake scorer), the calibration manifest and its split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.compressions import turboquant_mse
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.joint_kl import calibration_set
from engine.layer_select.joint_kl.scorer import merge
from engine.layer_select.joint_kl.search import SearchConfig, kl_mean, run_search
from engine.layer_select.rungs import TURBOQUANT_MSE
from engine.layer_select.slots import Slot, all_slots

DOMAINS = ("wikitext", "arc_5shot", "gsm8k", "mbpp")
BITS_COST = {4: 1.0, 3: 3.0, 2.5: 5.0, 2: 9.0, 1: 20.0}


def additive(assignment):
    """Separable fake KL: each slot adds weight x cost of its level; domains on different scales."""
    total = sum((1 + s.layer) * (2 if s.target == "k" else 1) * BITS_COST[lvl] for s, lvl in assignment.items())
    return {d: total * (i + 1) for i, d in enumerate(DOMAINS)}


def interacting(assignment):
    """Keys of adjacent layers hurt more together, so the joint order differs from isolated scores."""
    base = additive(assignment)
    keys = {s.layer for s in assignment if s.target == "k"}
    extra = sum(20.0 for layer in keys if layer + 1 in keys)
    return {d: v + extra for d, v in base.items()}


class Counting:
    def __init__(self, fn):
        self.fn, self.calls, self.maps = fn, 0, 0

    def __call__(self, candidates):
        self.calls += 1
        self.maps += len(candidates)
        return {key: self.fn(a) for key, a in candidates.items()}


def config(lazy_top=0, budgets=(0.70, 0.80, 0.85), max_rounds=None, n_layers=4):
    slots = all_slots(n_layers)
    return SearchConfig(method_kv=turboquant_mse(), rungs=TURBOQUANT_MSE, slots=slots, budgets=budgets,
                        reference={s: 3 for s in slots}, lazy_top=lazy_top, full_every=3, tag_prefix="t_",
                        max_rounds=max_rounds)


def quiet(*_args):
    return None


class SearchTests(unittest.TestCase):
    def test_kl_mean_divides_each_domain_by_its_reference(self):
        score, normed = kl_mean({"a": 2.0, "b": 1.0}, {"a": 4.0, "b": 1.0})
        self.assertEqual(normed, {"a": 0.5, "b": 1.0})
        self.assertAlmostEqual(score, 0.75)

    def test_budgets_are_reached_nested_and_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            snaps = run_search(Counting(interacting), config(), tmp, log=quiet)
            self.assertEqual([s["tag"] for s in snaps], ["t_p70", "t_p80", "t_p85"])
            index = {r.level: i for i, r in enumerate(TURBOQUANT_MSE)}
            previous = {}
            for snap in snaps:
                self.assertGreaterEqual(snap["compression"], snap["budget"] - 1e-12)
                parse_kv_spec(snap["kv"])
                levels = {Slot.from_dict(a): index[a["level"]] for a in snap["assignment"]}
                for slot, i in previous.items():
                    self.assertGreaterEqual(levels[slot], i)  # nested: no slot goes back down
                previous = levels
                self.assertTrue((Path(tmp) / f"selected_{snap['tag']}.json").exists())
            history = [json.loads(line) for line in (Path(tmp) / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), snaps[-1]["calibration"]["rounds"])
            self.assertTrue(all(h["kl_mean_after"] >= 0 for h in history))

    def test_first_rounds_take_cheapest_slots_to_4_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_search(Counting(additive), config(budgets=(0.70,)), tmp, log=quiet)
            history = [json.loads(line) for line in (Path(tmp) / "history.jsonl").read_text().splitlines()]
            self.assertEqual(history[0]["pick"], "v00")
            self.assertEqual(history[0]["pick_level"], 4)

    def test_resume_gives_the_same_maps(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            straight = run_search(Counting(interacting), config(lazy_top=3), a, log=quiet)
            run_search(Counting(interacting), config(lazy_top=3, max_rounds=5), b, log=quiet)
            resumed = run_search(Counting(interacting), config(lazy_top=3), b, log=quiet)
            self.assertEqual([s["assignment"] for s in straight], [s["assignment"] for s in resumed])

    def test_resume_refuses_other_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_search(Counting(additive), config(max_rounds=2), tmp, log=quiet)
            with self.assertRaises(ValueError):
                run_search(Counting(additive), config(lazy_top=5), tmp, log=quiet)

    def test_lazy_matches_full_when_slots_do_not_interact_and_scores_fewer_maps(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            full, lazy = Counting(additive), Counting(additive)
            snaps_full = run_search(full, config(lazy_top=0, n_layers=6), a, log=quiet)
            snaps_lazy = run_search(lazy, config(lazy_top=3, n_layers=6), b, log=quiet)
            self.assertEqual([s["assignment"] for s in snaps_full], [s["assignment"] for s in snaps_lazy])
            self.assertLess(lazy.maps, full.maps)

    def test_search_can_start_from_a_map_on_the_whole_bit_ladder(self):
        from engine.layer_select.rungs import TURBOQUANT_MSE_WHOLE

        slots = all_slots(4)
        start = {s: (3 if s.layer < 2 else 2) for s in slots}
        cfg = SearchConfig(method_kv=turboquant_mse(), rungs=TURBOQUANT_MSE_WHOLE, slots=slots, budgets=(0.90,),
                           reference={s: 3 for s in slots}, lazy_top=3, full_every=3, tag_prefix="w_", start=start)
        with tempfile.TemporaryDirectory() as tmp:
            snap = run_search(Counting(additive), cfg, tmp, log=quiet)[0]
            history = [json.loads(line) for line in (Path(tmp) / "history.jsonl").read_text().splitlines()]
        levels = {Slot.from_dict(a): a["level"] for a in snap["assignment"]}
        self.assertTrue(all(levels[s] <= start[s] for s in slots))  # only climbs from the start
        self.assertGreaterEqual(snap["compression"], 0.90 - 1e-12)
        start_saving = (4 * 13 / 16 + 4 * 14 / 16) / 8
        self.assertEqual(len(history), round((snap["compression"] - start_saving) * 8 * 16))  # one 1/16 step per round
        self.assertNotEqual(cfg.identity(), config().identity())
        self.assertIn("start", cfg.identity())

    def test_start_level_off_the_ladder_is_refused(self):
        slots = all_slots(2)
        cfg = SearchConfig(method_kv=turboquant_mse(), rungs=TURBOQUANT_MSE, slots=slots, budgets=(0.85,),
                           reference={s: 3 for s in slots}, start={slots[0]: 1})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_search(Counting(additive), cfg, tmp, log=quiet)

    def test_whole_bit_start_map_moves_2_5_to_3(self):
        from compression_topics.turboquant_mse.scripts import turboquant_mse_joint_kl_whole as whole

        start = whole.start_map()
        self.assertEqual(len(start), 64)
        self.assertTrue(set(start.values()) <= {4, 3, 2})
        first = whole.start_selection(start)
        self.assertAlmostEqual(first["compression"], (3 * 12 + 26 * 13 + 35 * 14) / 16 / 64)
        self.assertEqual({s["recent_window"] for s in first["kv"]["pipeline"]}, {128})
        run = whole.jkl.tasks_run([first, whole.uniform_2_bits()])
        self.assertEqual(len(run["configurations"]), 3 * 3)

    def test_budget_above_top_rung_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            run_search(Counting(additive), config(budgets=(0.9,)), tmp, log=quiet)


class CalibrationSetTests(unittest.TestCase):
    def test_manifest_matches_its_hash_and_has_128_per_domain(self):
        rows, meta = calibration_set.load()
        self.assertEqual(len(rows), 512)
        self.assertEqual({d: sum(r["domain"] == d for r in rows) for d in DOMAINS}, {d: 128 for d in DOMAINS})
        self.assertEqual(meta["model"], "meta-llama/Llama-3.1-8B-Instruct")

    def test_per_domain_limit(self):
        rows, _ = calibration_set.load(per_domain=2)
        self.assertEqual(len(rows), 8)

    def test_partition_covers_every_example_once_and_balances_length(self):
        rows, _ = calibration_set.load()
        shards = calibration_set.partition(rows, 4)
        self.assertEqual(sorted(r["position"] for s in shards for r in s), list(range(512)))
        loads = [sum(calibration_set.cost(r) for r in s) for s in shards]
        self.assertLess(max(loads) / min(loads), 1.01)

    def test_merge_adds_sums_and_counts_before_dividing(self):
        merged = merge([{"m": {"gsm8k": [1.0, 1]}}, {"m": {"gsm8k": [2.0, 3]}}])
        self.assertEqual(merged, {"m": {"gsm8k": 0.75}})


class SamplesTests(unittest.TestCase):
    def test_write_samples_keeps_one_line_per_question_and_filter(self):
        from engine.eval_runner.execute import write_samples

        results = {"samples": {"gsm8k": [
            {"doc_id": 1, "filter": "strict-match", "prompt_hash": "b", "metrics": ["exact_match"],
             "exact_match": 0.0, "filtered_resps": ["[invalid]"], "doc": {"big": "x"}},
            {"doc_id": 0, "filter": "flexible-extract", "prompt_hash": "a", "metrics": ["exact_match"],
             "exact_match": 1.0, "filtered_resps": ["18"]},
        ]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_samples({}, {"output_path": f"{tmp}/gsm8k_x.json"}, results)
            self.assertEqual(path.name, "gsm8k_x.samples.jsonl")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["doc_id"] for r in rows], [0, 1])
        self.assertEqual(rows[0], {"task": "gsm8k", "doc_id": 0, "filter": "flexible-extract", "prompt_hash": "a",
                                   "metrics": {"exact_match": 1.0}, "response": "18"})

    def test_budget_run_logs_samples_of_selections_not_dense(self):
        from engine.layer_select.budgets import budget_run

        selection = {"tag": "x", "kv": turboquant_mse(), "budget": 0.7, "compression": 0.75}
        run = budget_run([selection], log_samples=True)
        logged = {c["name"]: c.get("log_samples", False) for c in run["configurations"]}
        self.assertTrue(all(v for k, v in logged.items() if not k.endswith("_dense")))
        self.assertFalse(any(v for k, v in logged.items() if k.endswith("_dense")))
        self.assertFalse(any("log_samples" in c for c in budget_run([selection])["configurations"]))


class CombinedRunTests(unittest.TestCase):
    def test_both_arms_share_one_task_run_with_dense_once(self):
        from compression_topics.turboquant_mse.scripts.turboquant_mse_calibrate import tasks_run

        maps = [{"tag": f"{arm}p{b}", "kv": turboquant_mse(), "budget": b / 100, "compression": b / 100}
                for arm in ("turboquant_mse_rp_", "turboquant_mse_jkl_") for b in (70, 80, 85)]
        names = [c["name"] for c in tasks_run(maps)["configurations"]]
        outputs = [c["output_path"] for c in tasks_run(maps)["configurations"]]
        self.assertEqual(len(names), 3 * 7)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(len(set(outputs)), len(outputs))
        self.assertEqual(sum(n.endswith("_dense") for n in names), 3)
        self.assertTrue(all(n.endswith("_dense") for n in names[-3:]))
        self.assertIn("gsm8k", names[0])

    def test_calibration_uses_the_32_token_window_and_tasks_128(self):
        from compression_topics.turboquant_mse.scripts.turboquant_mse_joint_kl import task_kv
        from compression_topics.turboquant_mse.scripts.turboquant_mse_study import rope_pair_steps, sweep_run

        swept = [c["kv"]["pipeline"][0] for c in sweep_run()["configurations"] if (c.get("kv") or {}).get("pipeline")]
        self.assertTrue(swept and all(step["recent_window"] == 32 for step in swept))
        self.assertTrue(all(rope_pair_steps(step) for step in swept))
        step = turboquant_mse()["pipeline"][0]
        self.assertEqual(step["recent_window"], 128)
        self.assertFalse(rope_pair_steps(step))  # rows swept with the task window are not mixed in
        self.assertFalse(rope_pair_steps({k: v for k, v in swept[0].items() if k != "signs"}))

        with tempfile.TemporaryDirectory() as tmp:
            snap = run_search(Counting(additive), config(budgets=(0.70,)), tmp, log=quiet)[0]
        windowed = json.loads(json.dumps(snap).replace('"recent_window": 128', '"recent_window": 32'))
        fixed = task_kv(windowed)
        self.assertEqual({s["recent_window"] for s in fixed["kv"]["pipeline"]}, {128})
        self.assertEqual(parse_kv_spec(fixed["kv"]), parse_kv_spec(snap["kv"]))


if __name__ == "__main__":
    unittest.main()
