"""CPU tests for layer sweeps, search spaces, stepwise greedy, and apply."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.compressions import CHECKSPARSE_L1_50, SPARSIFY_48, checksparse_l1
from kv_compress.spec import parse_kv_spec
from layer_select.apply import kv_for_assignment, kv_for_slot, kv_for_slots, parse_singleton, slot_from_kv
from layer_select.greedy.rank_fill import rank_fill
from layer_select.greedy.sequential import run_sequential
from layer_select.levels import DEFAULT_LEVELS, next_level
from layer_select.scores import ScoreRow, load_sweep_scores, word_perplexity
from layer_select.slots import Slot, keys_only, mix, space_slots
from layer_select.sweep import expand_singleton_configs


LEVELS_2 = (15, 30)


class SlotTests(unittest.TestCase):
    def test_keys_only_and_mix_counts(self):
        self.assertEqual(len(keys_only(4)), 4)
        self.assertEqual(len(mix(4)), 8)
        self.assertEqual(space_slots("keys_only", 4)[0], Slot(0, "k"))
        self.assertIn(Slot(0, "v"), space_slots("mix", 4))
        self.assertNotIn(Slot(0, "v"), space_slots("keys_only", 4))

    def test_next_level_cannot_skip(self):
        self.assertEqual(next_level(0, DEFAULT_LEVELS), 15)
        self.assertEqual(next_level(15, DEFAULT_LEVELS), 30)
        self.assertEqual(next_level(60, DEFAULT_LEVELS), None)


class SweepExpanderTests(unittest.TestCase):
    def test_mix_is_dense_plus_slots_times_levels(self):
        configs = expand_singleton_configs(
            n_layers=4,
            space="mix",
            method_kv=CHECKSPARSE_L1_50,
            method_tag="checksparse",
            results_dir="results/sweep",
            name_prefix="m",
            levels=LEVELS_2,
        )
        self.assertEqual(len(configs), 1 + 8 * 2)
        self.assertEqual(configs[0]["kv"], {"pipeline": []})
        slot, level = parse_singleton(configs[1]["kv"])
        self.assertEqual(slot, Slot(0, "k"))
        self.assertEqual(level, 15)
        slot, level = parse_singleton(configs[2]["kv"])
        self.assertEqual((slot, level), (Slot(0, "k"), 30))

    def test_keys_only_is_dense_plus_l_times_levels(self):
        configs = expand_singleton_configs(
            n_layers=4,
            space="keys_only",
            method_kv=CHECKSPARSE_L1_50,
            method_tag="checksparse",
            results_dir="results/sweep",
            name_prefix="m",
            levels=LEVELS_2,
        )
        self.assertEqual(len(configs), 1 + 4 * 2)
        parsed = [parse_singleton(item["kv"]) for item in configs[1:]]
        self.assertEqual(parsed[0], (Slot(0, "k"), 15))
        self.assertTrue(all(slot.target == "k" for slot, _level in parsed))


def _row(slot, level, delta, dense=10.0):
    return ScoreRow(slot=slot, level=level, ppl=dense + delta, delta=delta, path=f"{slot.tag()}p{level}.json")


class RankFillTests(unittest.TestCase):
    def test_climbs_one_rung_and_ignores_v_in_keys_only(self):
        slots = keys_only(2)
        rows = [
            _row(Slot(0, "k"), 15, 2.0),
            _row(Slot(0, "k"), 30, 2.2),
            _row(Slot(1, "k"), 15, 0.2),
            _row(Slot(1, "k"), 30, 3.0),
            _row(Slot(0, "v"), 15, 0.01),
            _row(Slot(0, "v"), 30, 0.02),
        ]
        # 0.15 mean on 2 slots = 30 percentage-points: one 15% rung on each key.
        chosen = rank_fill(rows, slots, budget=0.15, levels=LEVELS_2, dense_ppl=10.0)
        self.assertEqual(chosen, {Slot(1, "k"): 15, Slot(0, "k"): 15})
        self.assertNotIn(Slot(0, "v"), chosen)

    def test_cannot_skip_first_rung(self):
        slots = keys_only(1)
        rows = [
            _row(Slot(0, "k"), 15, 10.0),
            _row(Slot(0, "k"), 30, 10.1),
        ]
        chosen = rank_fill(rows, slots, budget=0.15, levels=LEVELS_2, dense_ppl=10.0)
        self.assertEqual(chosen, {Slot(0, "k"): 15})


class ApplyTests(unittest.TestCase):
    def test_mixed_levels_parse_as_two_steps(self):
        assignment = {Slot(1, "k"): 15, Slot(3, "k"): 30, Slot(2, "v"): 15}
        kv = kv_for_assignment(checksparse_l1("all", "all"), assignment)
        spec = parse_kv_spec(kv)
        self.assertEqual(len(spec.pipeline), 2)
        by_pct = {step.kwargs["prune_pct"]: step for step in spec.pipeline}
        self.assertEqual(by_pct[15].k_layers, frozenset({1}))
        self.assertEqual(by_pct[15].v_layers, frozenset({2}))
        self.assertEqual(by_pct[30].k_layers, frozenset({3}))
        empty = parse_kv_spec(kv_for_assignment(CHECKSPARSE_L1_50, {}))
        self.assertTrue(empty.is_identity())

    def test_uniform_slots_still_work(self):
        kv = kv_for_slots(SPARSIFY_48, [Slot(1, "k"), Slot(2, "v")])
        spec = parse_kv_spec(kv)
        self.assertEqual(spec.pipeline[0].method, "sparsify_nm")
        self.assertEqual(slot_from_kv(kv_for_slot(CHECKSPARSE_L1_50, Slot(4, "v"), level=40)), Slot(4, "v"))


class SequentialTests(unittest.TestCase):
    def test_picks_cheapest_legal_rung_given_current_set(self):
        costs = {
            (Slot(0, "k"), 15): 5.0,
            (Slot(0, "k"), 30): 5.2,
            (Slot(1, "k"), 15): 0.5,
            (Slot(1, "k"), 30): 4.0,
        }

        def evaluate_assignment(assignment):
            return 10.0 + sum(costs[(slot, pct)] for slot, pct in assignment.items())

        chosen = run_sequential(
            slots=keys_only(2),
            budget=0.075,
            evaluate_assignment=evaluate_assignment,
            levels=LEVELS_2,
            dense_ppl=10.0,
        )
        self.assertEqual(chosen, {Slot(1, "k"): 15})


class ScoreReaderTests(unittest.TestCase):
    def test_load_sweep_scores_from_lm_eval_shape(self):
        def dump(path: Path, ppl: float, kv: dict):
            payload = {
                "results": {"wikitext": {"word_perplexity,none": ppl}},
                "configs": {"wikitext": {"metadata": {"kv": kv}}},
            }
            path.write_text(json.dumps(payload))

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dump(folder / "dense.json", 10.0, {"pipeline": []})
            dump(folder / "k0p15.json", 10.4, kv_for_slot(CHECKSPARSE_L1_50, Slot(0, "k"), level=15))
            dump(folder / "k0p30.json", 11.0, kv_for_slot(CHECKSPARSE_L1_50, Slot(0, "k"), level=30))
            dense_ppl, rows = load_sweep_scores(folder)
            self.assertEqual(dense_ppl, 10.0)
            by_key = {(row.slot, row.level): row.delta for row in rows}
            self.assertAlmostEqual(by_key[(Slot(0, "k"), 15)], 0.4)
            self.assertAlmostEqual(by_key[(Slot(0, "k"), 30)], 1.0)
            self.assertEqual(word_perplexity({"results": {"wikitext": {"word_perplexity,none": 3.5}}}), 3.5)


if __name__ == "__main__":
    unittest.main()
