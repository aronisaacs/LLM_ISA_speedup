"""Hadamard codebook rewrite and the WikiText run. No model load."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from catalog.compressions import turboquant_mse
from compression_topics.turboquant_mse.algorithms.turboquant_mse import (
    apply,
    centroids,
    hadamard,
    orthogonal,
    sign_diagonal,
)
from compression_topics.turboquant_mse.scripts.calibrate_outliers import ChannelStats
from compression_topics.turboquant_mse.scripts.turboquant_mse_levels import levels_run
from engine.kv_compress import parse_kv_spec
from engine.layer_select.apply import kv_for_assignment
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import next_level
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import Slot, all_slots
from compression_topics.turboquant_mse.scripts.plot_turboquant_mse import write_turboquant_mse_study
from compression_topics.turboquant_mse.scripts.turboquant_mse_study import sweep_run


class TurboquantMseRewriteTests(unittest.TestCase):
    def test_hadamard_is_its_own_inverse(self):
        values = torch.randn(2, 8)
        self.assertTrue(torch.allclose(hadamard(hadamard(values)), values, atol=1e-5))

    def test_one_bit_centroids_match_the_gaussian_mean(self):
        level = math.sqrt(2.0 / math.pi) / math.sqrt(4)
        self.assertTrue(torch.allclose(centroids(1, 4), torch.tensor([-level, level]), atol=1e-5))

    def test_rebuild_uses_the_rotated_codebook_and_fp16_norm(self):
        features = 8
        bits = 4
        token = torch.randn(features)
        rebuilt = apply(token.view(1, 1, 1, features), layer_idx=2, target="k", bits=bits).view(-1)
        norm = token.norm()
        stored = norm.to(dtype=torch.float16).float()
        signs = sign_diagonal(2, features)
        rotated = hadamard(token / norm * signs)
        levels = centroids(bits, features)
        snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
        expected = hadamard(snapped) * signs * stored
        self.assertTrue(torch.allclose(rebuilt, expected, atol=1e-5))

    def test_values_use_the_same_rebuild(self):
        token = torch.randn(1, 1, 1, 8)
        keys = apply(token, layer_idx=1, target="k", bits=4)
        values = apply(token, layer_idx=1, target="v", bits=4)
        self.assertTrue(torch.equal(keys, values))

    def test_zero_token_stays_zero(self):
        token = torch.zeros(1, 1, 2, 4)
        self.assertTrue(torch.equal(apply(token, layer_idx=0, target="k", bits=4), token))


class TurboquantMseRunTests(unittest.TestCase):
    def test_sweep_scores_every_slot_at_each_bit_width(self):
        configurations = sweep_run()["configurations"]
        names = [item["name"] for item in configurations]
        self.assertEqual(names[0], "llama31_dense")
        self.assertEqual(len(names), 1 + 32 * 2 * 4)
        four = next(item for item in configurations if item["name"] == "llama31_turboquant_mse_k00_p4")
        one = next(item for item in configurations if item["name"] == "llama31_turboquant_mse_k00_p1")
        self.assertEqual(four["kv"]["pipeline"][0]["method"], "turboquant_mse")
        self.assertEqual(four["kv"]["pipeline"][0]["bits"], 4)
        self.assertEqual(one["kv"]["pipeline"][0]["bits"], 1)
        self.assertEqual(four["tasks"], ["wikitext"])

    def test_climber_walks_four_bits_down_to_one(self):
        widths = tuple(rung.level for rung in rungs_for("turboquant_mse"))
        self.assertEqual(widths, (4, 3, 2, 1))
        self.assertEqual([next_level(0, widths), next_level(4, widths), next_level(1, widths)], [4, 3, None])

    def test_assignment_can_mix_four_bits_and_one_bit(self):
        kv = kv_for_assignment(turboquant_mse(), {Slot(3, "k"): 4, Slot(7, "v"): 1})
        by_bits = {step["bits"]: step for step in kv["pipeline"]}
        self.assertEqual(by_bits[4]["k_layers"], [3])
        self.assertEqual(by_bits[4]["v_layers"], [])
        self.assertEqual(by_bits[1]["k_layers"], [])
        self.assertEqual(by_bits[1]["v_layers"], [7])

    def test_greedy_leaves_one_slot_at_four_bits_and_climbs_another_to_one(self):
        levels = tuple(rung.level for rung in rungs_for("turboquant_mse"))
        rows = []
        for slot in all_slots(1):
            for level in levels:
                rows.append(ScoreRow(slot=slot, level=level, ppl=60.0, delta=50.0, path=f"{slot.tag()}p{level}.json"))
        cheap = {
            (Slot(0, "v"), 4): 0.01,
            (Slot(0, "k"), 4): 0.05,
            (Slot(0, "k"), 3): 0.06,
            (Slot(0, "k"), 2): 0.07,
            (Slot(0, "k"), 1): 0.08,
        }
        for (slot, level), delta in cheap.items():
            rows = [row for row in rows if not (row.slot == slot and row.level == level)]
            rows.append(ScoreRow(slot=slot, level=level, ppl=10.0 + delta, delta=delta, path=f"{slot.tag()}p{level}.json"))
        chosen = rank_fill(rows, n_layers=1, budget=0.84, dense_ppl=10.0, rungs=rungs_for("turboquant_mse"))
        self.assertEqual(chosen, {Slot(0, "v"): 4, Slot(0, "k"): 1})

    def test_figures_cover_each_bit_width_and_each_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_turboquant_mse_study(tmp)
            names = sorted(path.name for path in paths)
            gsm8k = Path(tmp, "turboquant_mse_gsm8k.svg").read_text()
        self.assertEqual(
            names,
            [
                "turboquant_mse_ceval.svg",
                "turboquant_mse_gsm8k.svg",
                "turboquant_mse_humaneval.svg",
                "turboquant_mse_sweep_1bit.svg",
                "turboquant_mse_sweep_2bit.svg",
                "turboquant_mse_sweep_3bit.svg",
                "turboquant_mse_sweep_4bit.svg",
            ],
        )
        self.assertIn("Baseline", gsm8k)
        self.assertIn("60%", gsm8k)
        self.assertIn("90%", gsm8k)


def _write_table(directory, heads, features, ranking):
    """Calibration table with the same ranking for every layer, head and target."""
    path = Path(directory, "table.json")
    per_layer = [[list(ranking) for _ in range(heads)] for _ in range(4)]
    path.write_text(json.dumps({"ranking_length": features, "channels": {"k": per_layer, "v": per_layer}}))
    return str(path)


def _reference(part, layer_idx, bits):
    """TurboQuant of one part by hand: Hadamard path for powers of two, random orthogonal otherwise."""
    features = part.shape[-1]
    norm = part.norm(dim=-1, keepdim=True)
    stored = norm.to(dtype=torch.float16).float()
    unit = part / norm
    levels = centroids(bits, features)
    if features & (features - 1):
        rotation = orthogonal(layer_idx, features)
        rotated = unit @ rotation.T
        snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
        return snapped @ rotation * stored
    signs = sign_diagonal(layer_idx, features)
    rotated = hadamard(unit * signs)
    snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
    return hadamard(snapped) * signs * stored


class TurboquantMseOutlierSplitTests(unittest.TestCase):
    def test_plain_runs_keep_their_kv_options(self):
        self.assertEqual(turboquant_mse(bits=3)["pipeline"][0],
                         {"method": "turboquant_mse", "k_layers": "all", "v_layers": "all", "bits": 3})

    def test_orthogonal_is_orthogonal_and_seeded(self):
        rotation = orthogonal(5, 96)
        self.assertTrue(torch.allclose(rotation @ rotation.T, torch.eye(96), atol=1e-5))
        self.assertTrue(torch.equal(rotation, orthogonal(5, 96)))
        self.assertFalse(torch.equal(rotation, orthogonal(6, 96)))

    def test_split_quantizes_outliers_and_the_rest_as_two_vectors(self):
        heads, features, layer = 2, 8, 1
        ranking = [6, 1, 0, 2, 3, 4, 5, 7]   # outliers: channels 6 and 1
        tensor = torch.randn(1, heads, 3, features)
        with tempfile.TemporaryDirectory() as tmp:
            table = _write_table(tmp, heads, features, ranking)
            rebuilt = apply(tensor, layer_idx=layer, target="k", bits=2, outliers=2, outlier_table=table)
        high, low = [1, 6], [0, 2, 3, 4, 5, 7]
        self.assertTrue(torch.allclose(rebuilt[..., high], _reference(tensor[..., high], layer, 3), atol=1e-5))
        self.assertTrue(torch.allclose(rebuilt[..., low], _reference(tensor[..., low], layer, 2), atol=1e-5))

    def test_outlier_bits_overrides_bits_plus_one(self):
        tensor = torch.randn(1, 1, 2, 8)
        with tempfile.TemporaryDirectory() as tmp:
            table = _write_table(tmp, 1, 8, range(8))
            rebuilt = apply(tensor, layer_idx=0, target="v", bits=2, outliers=4, outlier_bits=4, outlier_table=table)
        self.assertTrue(torch.allclose(rebuilt[..., :4], _reference(tensor[..., :4], 0, 4), atol=1e-5))

    def test_split_options_are_checked_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = _write_table(tmp, 1, 8, range(8))
            parse_kv_spec(turboquant_mse(bits=2, outliers=4, outlier_table=table))
            with self.assertRaises(ValueError):
                parse_kv_spec(turboquant_mse(bits=2, outliers=8, outlier_table=table))
        with self.assertRaises(ValueError):
            parse_kv_spec({"pipeline": [{"method": "turboquant_mse", "k_layers": "all", "v_layers": "all",
                                         "bits": 2, "outliers": 4}]})

    def test_calibration_ranks_channels_by_mean_magnitude(self):
        stats = ChannelStats()
        for layer in range(2):
            tensor = torch.zeros(1, 2, 5, 4)
            tensor[:, 0, :, 2] = -3.0
            tensor[:, 0, :, 0] = 1.0
            tensor[:, 1, :, 3] = 2.0
            stats.add(layer, "k", tensor)
            stats.add(layer, "v", tensor * 0.5)
        table = stats.table({})
        self.assertEqual(table["ranking_length"], 4)
        self.assertEqual(table["channels"]["k"][1][0][:2], [2, 0])
        self.assertEqual(table["channels"]["v"][0][1][0], 3)
        self.assertAlmostEqual(table["mean_abs"]["k"][0][0][2], 3.0)

    def test_levels_run_has_three_levels_on_three_tasks_plus_dense(self):
        configurations = levels_run()["configurations"]
        self.assertEqual(len(configurations), 3 * 4)
        # Longest first: each of 3 GPUs (every 3rd configuration) gets one GSM8K, HumanEval and CEval run.
        self.assertEqual([c["tasks"][0] for c in configurations[:9:3]], ["gsm8k", "humaneval_instruct", "ceval-valid"])
        self.assertTrue(all(c["name"].endswith("_dense") for c in configurations[9:]))
        split = [c for c in configurations if "out32_3b_2b" in c["name"]]
        self.assertEqual(len(split), 3)
        step = split[0]["kv"]["pipeline"][0]
        self.assertEqual((step["bits"], step["outliers"]), (2, 32))
        self.assertEqual(split[0]["metadata"]["kv_compression"], 1 - (32 * 3 + 96 * 2) / 128 / 16)  # 2.25 bits
        plain = {c["metadata"]["kv_compression"] for c in configurations if "all3" in c["name"] or "all2_" in c["name"]
                 or c["name"].endswith("all2")}
        self.assertEqual(plain, {1 - 3 / 16, 1 - 2 / 16})


if __name__ == "__main__":
    unittest.main()
