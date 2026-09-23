"""CPU tests for the 25/50/75 key-and-value sweep, greedy rung climb, and apply."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catalog.compressions import CHECKSPARSE_L1_50, SPARSIFY_48, checksparse_l1
from kv_compress.spec import parse_kv_spec
from layer_select.apply import kv_for_assignment, kv_for_slot, kv_for_slots, parse_singleton, slot_from_kv
from layer_select.greedy.rank_fill import rank_fill
from layer_select.levels import LEVELS, next_level
from layer_select.scores import ScoreRow, load_sweep_scores, word_perplexity
from layer_select.slots import Slot, all_slots
from layer_select.sweep import expand_singleton_configs


class SlotTests(unittest.TestCase):
    def test_every_layer_has_a_key_and_a_value(self):
        slots = all_slots(4)
        self.assertEqual(len(slots), 8)
        self.assertEqual(slots[0], Slot(0, "k"))
        self.assertEqual(slots[1], Slot(0, "v"))
        self.assertIn(Slot(3, "v"), slots)

    def test_next_level_is_25_then_50_then_75(self):
        self.assertEqual(LEVELS, (25, 50, 75))
        self.assertEqual(next_level(0), 25)
        self.assertEqual(next_level(25), 50)
        self.assertEqual(next_level(50), 75)
        self.assertEqual(next_level(75), None)


class SweepExpanderTests(unittest.TestCase):
    def test_dense_plus_each_key_and_value_at_each_rung(self):
        configs = expand_singleton_configs(
            n_layers=2,
            method_kv=CHECKSPARSE_L1_50,
            method_tag="checksparse",
            results_dir="results/sweep",
            name_prefix="m",
        )
        self.assertEqual(len(configs), 1 + 4 * 3)
        self.assertEqual(configs[0]["kv"], {"pipeline": []})
        parsed = [parse_singleton(item["kv"]) for item in configs[1:]]
        self.assertEqual(
            parsed[:4],
            [
                (Slot(0, "k"), 25),
                (Slot(0, "k"), 50),
                (Slot(0, "k"), 75),
                (Slot(0, "v"), 25),
            ],
        )
        self.assertEqual({slot.target for slot, _level in parsed}, {"k", "v"})
        self.assertEqual({level for _slot, level in parsed}, {25, 50, 75})


def _row(slot, level, delta, dense=10.0):
    return ScoreRow(slot=slot, level=level, ppl=dense + delta, delta=delta, path=f"{slot.tag()}p{level}.json")


def _table(n_layers, overrides, dense=10.0):
    rows = []
    for slot in all_slots(n_layers):
        for level in LEVELS:
            delta = overrides.get((slot, level), 100.0)
            rows.append(_row(slot, level, delta, dense))
    return rows


class RankFillTests(unittest.TestCase):
    def test_climbs_the_cheaper_tensor_one_rung_at_a_time(self):
        rows = _table(
            1,
            {
                (Slot(0, "v"), 25): 0.1,
                (Slot(0, "v"), 50): 0.3,
                (Slot(0, "v"), 75): 5.0,
                (Slot(0, "k"), 25): 2.0,
                (Slot(0, "k"), 50): 2.2,
                (Slot(0, "k"), 75): 2.3,
            },
        )
        # Two +25 steps on 2 slots: 50 / 200 = 0.25.
        chosen = rank_fill(rows, n_layers=1, budget=0.25, dense_ppl=10.0)
        self.assertEqual(chosen, {Slot(0, "v"): 50})

    def test_cannot_jump_to_a_higher_rung(self):
        rows = _table(
            1,
            {
                (Slot(0, "k"), 25): 5.0,
                (Slot(0, "k"), 50): 5.05,
                (Slot(0, "k"), 75): 0.05,
                (Slot(0, "v"), 25): 1.0,
                (Slot(0, "v"), 50): 4.0,
                (Slot(0, "v"), 75): 8.0,
            },
        )
        chosen = rank_fill(rows, n_layers=1, budget=0.125, dense_ppl=10.0)
        self.assertEqual(chosen, {Slot(0, "v"): 25})

    def test_can_take_a_key_and_a_value_on_different_layers(self):
        rows = _table(
            2,
            {
                (Slot(0, "k"), 25): 0.2,
                (Slot(1, "v"), 25): 0.3,
            },
        )
        chosen = rank_fill(rows, n_layers=2, budget=0.125, dense_ppl=10.0)
        self.assertEqual(chosen, {Slot(0, "k"): 25, Slot(1, "v"): 25})


class ApplyTests(unittest.TestCase):
    def test_mixed_levels_parse_as_two_steps(self):
        assignment = {Slot(1, "k"): 25, Slot(3, "k"): 50, Slot(2, "v"): 25}
        kv = kv_for_assignment(checksparse_l1("all", "all"), assignment)
        spec = parse_kv_spec(kv)
        self.assertEqual(len(spec.pipeline), 2)
        by_pct = {step.kwargs["prune_pct"]: step for step in spec.pipeline}
        self.assertEqual(by_pct[25].k_layers, frozenset({1}))
        self.assertEqual(by_pct[25].v_layers, frozenset({2}))
        self.assertEqual(by_pct[50].k_layers, frozenset({3}))
        empty = parse_kv_spec(kv_for_assignment(CHECKSPARSE_L1_50, {}))
        self.assertTrue(empty.is_identity())

    def test_uniform_slots_still_work(self):
        kv = kv_for_slots(SPARSIFY_48, [Slot(1, "k"), Slot(2, "v")])
        spec = parse_kv_spec(kv)
        self.assertEqual(spec.pipeline[0].method, "sparsify_nm")
        self.assertEqual(slot_from_kv(kv_for_slot(CHECKSPARSE_L1_50, Slot(4, "v"), level=50)), Slot(4, "v"))


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
            dump(folder / "k0p25.json", 10.4, kv_for_slot(CHECKSPARSE_L1_50, Slot(0, "k"), level=25))
            dump(folder / "k0p50.json", 11.0, kv_for_slot(CHECKSPARSE_L1_50, Slot(0, "k"), level=50))
            dense_ppl, rows = load_sweep_scores(folder)
            self.assertEqual(dense_ppl, 10.0)
            by_key = {(row.slot, row.level): row.delta for row in rows}
            self.assertAlmostEqual(by_key[(Slot(0, "k"), 25)], 0.4)
            self.assertAlmostEqual(by_key[(Slot(0, "k"), 50)], 1.0)
            self.assertEqual(word_perplexity({"results": {"wikitext": {"word_perplexity,none": 3.5}}}), 3.5)


if __name__ == "__main__":
    unittest.main()
