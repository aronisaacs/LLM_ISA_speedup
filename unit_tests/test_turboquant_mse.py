"""Hadamard codebook rewrite and the WikiText run. No model load."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE, turboquant_mse
from compression_topics.turboquant_mse.algorithms.turboquant_mse import (
    apply,
    centroids,
    hadamard,
    orthogonal,
    rope_pair_signs,
    sign_diagonal,
)
from compression_topics.turboquant_mse.scripts.calibrate_outliers import ChannelStats
from compression_topics.turboquant_mse.scripts.turboquant_mse_levels import levels_run
from compression_topics.turboquant_mse.scripts.turboquant_mse_longbench import longbench_run, summary
from engine.kv_compress import parse_kv_spec
from engine.layer_select.apply import kv_for_assignment, kv_for_slot, parse_singleton
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
        by_name = {item["name"]: item["kv"]["pipeline"][0] for item in configurations[1:]}
        four = by_name["llama31_turboquant_mse_rp_k00_p4"]
        half = by_name["llama31_turboquant_mse_rp_k00_p2_5"]
        two = by_name["llama31_turboquant_mse_rp_v31_p2"]
        self.assertEqual((four["method"], four["bits"], four["signs"]), ("turboquant_mse", 4, "rope_pairs"))
        self.assertNotIn("outliers", four)
        self.assertEqual((half["bits"], half["outliers"], half["outlier_table"]), (2, 64, TURBOQUANT_OUTLIER_TABLE))
        self.assertEqual(two["bits"], 2)
        self.assertEqual(configurations[1]["tasks"], ["wikitext"])

    def test_climber_walks_four_bits_down_to_two(self):
        widths = tuple(rung.level for rung in rungs_for("turboquant_mse"))
        self.assertEqual(widths, (4, 3, 2.5, 2))
        self.assertEqual([next_level(0, widths), next_level(3, widths), next_level(2.5, widths), next_level(2, widths)],
                         [4, 2.5, 2, None])

    def test_half_bit_level_is_the_64_outlier_split_and_parses_back(self):
        kv = kv_for_slot(turboquant_mse(), Slot(5, "v"), level=2.5)
        step = kv["pipeline"][0]
        self.assertEqual((step["bits"], step["outliers"], step["outlier_table"]), (2, 64, TURBOQUANT_OUTLIER_TABLE))
        self.assertEqual(parse_singleton(kv), (Slot(5, "v"), 2.5))
        whole = kv_for_slot(kv, Slot(5, "v"), level=3)["pipeline"][0]
        self.assertEqual(whole["bits"], 3)
        self.assertNotIn("outliers", whole)
        self.assertNotIn("outlier_table", whole)
        self.assertEqual(parse_singleton(kv_for_slot(turboquant_mse(), Slot(0, "k"), level=4)), (Slot(0, "k"), 4))

    def test_assignment_can_mix_four_bits_and_the_half_bit_split(self):
        kv = kv_for_assignment(turboquant_mse(), {Slot(3, "k"): 4, Slot(7, "v"): 2.5, Slot(8, "k"): 2})
        by_level = {(step["bits"], step.get("outliers", 0)): step for step in kv["pipeline"]}
        self.assertEqual((by_level[(4, 0)]["k_layers"], by_level[(4, 0)]["v_layers"]), ([3], []))
        self.assertEqual((by_level[(2, 64)]["k_layers"], by_level[(2, 64)]["v_layers"]), ([], [7]))
        self.assertEqual((by_level[(2, 0)]["k_layers"], by_level[(2, 0)]["v_layers"]), ([8], []))

    def test_greedy_leaves_one_slot_at_four_bits_and_climbs_another_to_two(self):
        levels = tuple(rung.level for rung in rungs_for("turboquant_mse"))
        rows = []
        for slot in all_slots(1):
            for level in levels:
                rows.append(ScoreRow(slot=slot, level=level, ppl=60.0, delta=50.0, path=f"{slot.tag()}p{level}.json"))
        cheap = {
            (Slot(0, "v"), 4): 0.01,
            (Slot(0, "k"), 4): 0.05,
            (Slot(0, "k"), 3): 0.06,
            (Slot(0, "k"), 2.5): 0.065,
            (Slot(0, "k"), 2): 0.07,
        }
        for (slot, level), delta in cheap.items():
            rows = [row for row in rows if not (row.slot == slot and row.level == level)]
            rows.append(ScoreRow(slot=slot, level=level, ppl=10.0 + delta, delta=delta, path=f"{slot.tag()}p{level}.json"))
        chosen = rank_fill(rows, n_layers=1, budget=0.81, dense_ppl=10.0, rungs=rungs_for("turboquant_mse"))
        self.assertEqual(chosen, {Slot(0, "v"): 4, Slot(0, "k"): 2})

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


def _reference(part, layer_idx, bits, signs=None):
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
    signs = sign_diagonal(layer_idx, features) if signs is None else signs
    rotated = hadamard(unit * signs)
    snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
    return hadamard(snapped) * signs * stored


class TurboquantMseFastPathTests(unittest.TestCase):
    """The matrix path against the algorithm written out step by step, one vector at a time."""

    def _reference(self, x, layer, target, bits, outliers):
        from compression_topics.turboquant_mse.algorithms.turboquant_mse import outlier_channels

        d = x.shape[-1]
        flipped = x.double() * rope_pair_signs(layer, d).double()
        if outliers:
            channels = outlier_channels(TURBOQUANT_OUTLIER_TABLE, layer, target, outliers)
        out = torch.empty_like(flipped)
        for head in range(x.shape[1]):
            if outliers:
                high = sorted(channels[head].tolist())
                parts = [(high, bits + 1), ([c for c in range(d) if c not in high], bits)]
            else:
                parts = [(list(range(d)), bits)]
            for index, part_bits in parts:
                part = flipped[0, head, :, index]
                norm = part.norm(dim=-1, keepdim=True)
                n = len(index)
                if n & (n - 1):
                    rotation = orthogonal(layer, n).double()
                    rotated = (part / norm) @ rotation.T
                else:
                    rotated = hadamard(part / norm).double()
                levels = centroids(part_bits, n).double()
                snapped = levels[(rotated.unsqueeze(-1) - levels).abs().argmin(-1)]
                back = snapped @ rotation if n & (n - 1) else hadamard(snapped).double()
                out[0, head, :, index] = back * norm.half().double()
        return (out * rope_pair_signs(layer, d).double()).to(x.dtype)

    def test_matches_the_step_by_step_algorithm(self):
        torch.manual_seed(3)
        x = torch.randn(1, 8, 50, 128, dtype=torch.bfloat16)
        for bits, outliers in ((2, 0), (3, 0), (2, 64), (2, 32)):
            for layer, target in ((0, "k"), (13, "v")):
                kw = dict(bits=bits, signs="rope_pairs")
                if outliers:
                    kw.update(outliers=outliers, outlier_table=TURBOQUANT_OUTLIER_TABLE)
                fast = apply(x, layer_idx=layer, target=target, **kw).float().reshape(-1, 128)
                slow = self._reference(x, layer, target, bits, outliers).float().reshape(-1, 128)
                relative = (fast - slow).norm(dim=-1) / slow.norm(dim=-1)
                # A coordinate at a codebook midpoint may round either way; allow 1 % of vectors.
                self.assertLess((relative > 1e-2).float().mean().item(), 0.01, (bits, outliers, layer))
                self.assertLess(relative.median().item(), 1e-5, (bits, outliers, layer))


class TurboquantMseLongbenchTests(unittest.TestCase):
    def test_run_list(self):
        names = [c["name"] for c in longbench_run()["configurations"]]
        self.assertEqual(len(names), 10)
        self.assertTrue(names[0].startswith("llama31_longbench_e_tr_chat_"))
        self.assertEqual(sum("_dense" in n for n in names), 2)
        for c in longbench_run()["configurations"]:
            spec = parse_kv_spec(c["kv"])
            self.assertEqual(c["batch_size"], 1)
            for step in spec.pipeline:
                self.assertEqual(step.kwargs["recent_window"], 128)

    def test_paper_variant(self):
        paper = next(c for c in longbench_run()["configurations"] if c["name"].endswith("_paper_out32"))
        step = paper["kv"]["pipeline"][0]
        self.assertEqual((step["outliers"], step["rotation"], step["codebook"]), (32, "orthogonal", "beta"))
        self.assertEqual(paper["kv"]["prompt_attention"], "dense")
        self.assertAlmostEqual(paper["metadata"]["kv_compression"], 1 - 2.25 / 16)

    def test_untruncated_rows_keep_their_identity(self):
        from compression_topics.turboquant_mse.scripts.turboquant_mse_longbench import untruncated_selections

        for selection in untruncated_selections():
            self.assertNotIn("recent_window", selection["kv"]["pipeline"][0])

    def test_timing_run_is_limited_and_kept_out_of_results(self):
        configurations = longbench_run(limit=5)["configurations"]
        self.assertEqual(len(configurations), 4)
        for c in configurations:
            self.assertEqual(c["limit"], 5)
            self.assertTrue(c["output_path"].startswith("logs/longbench_timing/"))

    def test_summary_averages_categories_and_tasks(self):
        from catalog.tasks import LONGBENCH_E_TR_CHAT, LONGBENCH_E_TR_PLAIN

        rows = []
        for task in (LONGBENCH_E_TR_CHAT, LONGBENCH_E_TR_PLAIN):
            scores = {name: {"score,none": 0.5} for name in task["tasks"]}
            rows.append({"identity": {"pretrained": "meta-llama/Llama-3.1-8B-Instruct", "tasks": task["tasks"],
                                      "kv": {"pipeline": []}, "limit": None}, "scores": scores})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps({"simulations": rows}))
            text = summary(path).split("-- truncated")[1]
            line = next(l for l in text.splitlines() if l.startswith("ours dense"))
        self.assertEqual(line.split()[2:], ["50.00"] * 7)


class TurboquantMsePaperOptionsTests(unittest.TestCase):
    def test_beta_levels_approach_the_gaussian_ones_in_high_dimension(self):
        from compression_topics.turboquant_mse.algorithms.turboquant_mse import beta_centroids

        for bits in (2, 3):
            gaussian = centroids(bits, 1024)
            beta = beta_centroids(bits, 1024)
            self.assertLess(((beta - gaussian).abs() / gaussian.abs()).max().item(), 0.01)
            small = beta_centroids(bits, 32)
            self.assertTrue(torch.allclose(small, -small.flip(0), atol=1e-6))
            self.assertTrue(bool((small[1:] > small[:-1]).all()))
            self.assertLess(small.abs().max().item(), 1.0)

    def test_paper_known_two_bit_levels(self):
        from compression_topics.turboquant_mse.algorithms.turboquant_mse import beta_centroids

        # Paper, Section 3.1: b = 2 gives ±0.453/sqrt(d) and ±1.51/sqrt(d) for moderately large d.
        scaled = beta_centroids(2, 512) * math.sqrt(512)
        self.assertTrue(torch.allclose(scaled[2:], torch.tensor([0.453, 1.51]), atol=0.01))

    def test_orthogonal_rotation_is_orthogonal_and_differs_from_hadamard(self):
        from compression_topics.turboquant_mse.algorithms.turboquant_mse import _plan

        dense = _plan(0, "k", 2, 32, 3, TURBOQUANT_OUTLIER_TABLE, "rope_pairs", 8, 128, torch.device("cpu"),
                      "orthogonal", "beta")
        ours = _plan(0, "k", 2, 32, 3, TURBOQUANT_OUTLIER_TABLE, "rope_pairs", 8, 128, torch.device("cpu"))
        eye = torch.eye(128)
        self.assertTrue(torch.allclose(dense.matrix[0] @ dense.matrix[0].T, eye, atol=1e-5))
        self.assertFalse(torch.allclose(dense.matrix, ours.matrix))

    def test_options_are_validated_and_written_only_when_used(self):
        self.assertNotIn("rotation", turboquant_mse()["pipeline"][0])
        self.assertEqual(turboquant_mse()["pipeline"][0]["recent_window"], 128)
        self.assertNotIn("recent_window", turboquant_mse(recent_window=0)["pipeline"][0])
        with self.assertRaises(ValueError):
            parse_kv_spec(turboquant_mse(rotation="qr"))
        with self.assertRaises(ValueError):
            parse_kv_spec(turboquant_mse(codebook="uniform"))

    def test_dynamic_outliers_pick_the_largest_channels_of_each_group(self):
        torch.manual_seed(0)
        tensor = torch.randn(1, 2, 64, 128)
        tensor[:, :, :32, :32] *= 20  # group 0: channels 0-31 dominate
        tensor[:, :, 32:, 96:] *= 20  # group 1: channels 96-127
        options = dict(layer_idx=0, target="k", bits=2, outliers=32, rotation="orthogonal", codebook="beta",
                       signs="rope_pairs")
        grouped = apply(tensor, outlier_group=32, **options)
        halves = torch.cat([apply(tensor[:, :, :32], outlier_group=32, **options),
                            apply(tensor[:, :, 32:], outlier_group=32, **options)], dim=-2)
        self.assertTrue(torch.allclose(grouped, halves, atol=1e-6))  # groups do not interact
        # The dominant channels get the 3-bit part: their error is lower than with the outliers elsewhere.
        swapped = torch.cat([tensor[:, :, 32:], tensor[:, :, :32]], dim=-2)
        mixed = apply(torch.cat([tensor[:, :, :16], swapped[:, :, :16]], dim=-2), outlier_group=32, **options)
        error = lambda out, ref: ((out - ref).norm() / ref.norm()).item()
        self.assertLess(error(grouped, tensor), error(mixed, torch.cat([tensor[:, :, :16], swapped[:, :, :16]], dim=-2)))
        with self.assertRaises(ValueError):
            apply(tensor[:, :, :40], outlier_group=32, **options)

    def test_dynamic_outliers_need_a_window_and_age_out_in_groups(self):
        from transformers.cache_utils import DynamicCache
        from engine.kv_compress.cache import patch_cache_update

        with self.assertRaises(ValueError):
            parse_kv_spec(turboquant_mse(bits=2, outliers=32, outlier_group=32, recent_window=0))
        kv = turboquant_mse(bits=2, outliers=32, outlier_group=32, recent_window=4)
        self.assertNotIn("outlier_table", kv["pipeline"][0])
        uninstall = patch_cache_update(parse_kv_spec(kv))
        try:
            torch.manual_seed(0)
            cache = DynamicCache()
            tokens = [torch.randn(1, 2, 40, 128)] + [torch.randn(1, 2, 1, 128) for _ in range(28)]
            out_key, _ = cache.update(tokens[0].clone(), tokens[0].clone(), 0)
            self.assertTrue(torch.equal(out_key[..., 32:, :], tokens[0][..., 32:, :]))  # 36 aged: one whole group
            self.assertFalse(torch.allclose(out_key[..., :32, :], tokens[0][..., :32, :]))
            for step in tokens[1:]:
                out_key, _ = cache.update(step.clone(), step.clone(), 0)
            every = torch.cat(tokens, dim=-2)  # 68 tokens, 64 aged: two groups
            self.assertTrue(torch.equal(out_key[..., 64:, :], every[..., 64:, :]))
            self.assertTrue(torch.equal(out_key[..., 32:64, :],
                                        apply(every[..., 32:64, :], layer_idx=0, target="k", bits=2, outliers=32,
                                              outlier_group=32, signs="rope_pairs")))
        finally:
            uninstall()

    def test_truncation_keeps_both_ends(self):
        import importlib.util
        from unittest import mock

        spec = importlib.util.spec_from_file_location(
            "tr_prompts", Path(__file__).resolve().parents[1] / "catalog/lm_eval_tasks/longbench_tr/tr_prompts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Words:
            def __call__(self, text, truncation=False):
                return type("T", (), {"input_ids": text.split(" ")})()

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(ids) + "|"

        with mock.patch.object(module, "_tokenizer", lambda: Words()):
            self.assertEqual(module.truncate("a b c", maxlen=4), "a b c")
            self.assertEqual(module.truncate("a b c d e f", maxlen=4), "a b|e f|")


class TurboquantMseOutlierSplitTests(unittest.TestCase):
    def test_independent_signs_keep_the_earlier_kv_options(self):
        self.assertEqual(turboquant_mse(bits=3, signs="independent", recent_window=0)["pipeline"][0],
                         {"method": "turboquant_mse", "k_layers": "all", "v_layers": "all", "bits": 3})
        self.assertEqual(turboquant_mse(bits=3)["pipeline"][0]["signs"], "rope_pairs")

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

    def test_rope_pair_signs_commute_with_rope(self):
        signs = rope_pair_signs(3, 128)
        self.assertTrue(torch.equal(signs[:64], signs[64:]))
        self.assertTrue(torch.equal(signs[:64], sign_diagonal(3, 64)))
        x = torch.randn(5, 128)
        angle = torch.rand(5, 64) * 6.0
        cos, sin = torch.cat((angle.cos(), angle.cos()), -1), torch.cat((angle.sin(), angle.sin()), -1)
        rope = lambda t: t * cos + torch.cat((-t[..., 64:], t[..., :64]), -1) * sin   # HF rotate_half
        self.assertTrue(torch.allclose(rope(x * signs), rope(x) * signs, atol=1e-6))

    def test_rope_pair_signs_without_split(self):
        token = torch.randn(1, 1, 3, 16)
        rebuilt = apply(token, layer_idx=2, target="k", bits=3, signs="rope_pairs")
        self.assertTrue(torch.allclose(rebuilt, _reference(token, 2, 3, signs=rope_pair_signs(2, 16)), atol=1e-5))

    def test_rope_pair_signs_flip_original_channels_before_the_split(self):
        heads, features, layer = 1, 8, 2
        ranking = [6, 1, 0, 5, 2, 3, 4, 7]   # outliers: channels 6, 1, 0, 5
        tensor = torch.randn(1, heads, 3, features)
        with tempfile.TemporaryDirectory() as tmp:
            table = _write_table(tmp, heads, features, ranking)
            rebuilt = apply(tensor, layer_idx=layer, target="k", bits=2, outliers=4, outlier_table=table,
                            signs="rope_pairs")
        d = rope_pair_signs(layer, features)
        flipped = tensor * d
        for part, bits in (([0, 1, 5, 6], 3), ([2, 3, 4, 7], 2)):
            expected = _reference(flipped[..., part], layer, bits, signs=torch.ones(4)) * d[part]
            self.assertTrue(torch.allclose(rebuilt[..., part], expected, atol=1e-5))

    def test_unknown_sign_scheme_is_rejected_before_model_load(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"pipeline": [{"method": "turboquant_mse", "k_layers": "all", "v_layers": "all",
                                         "bits": 2, "signs": "pairs"}]})

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

    def test_levels_run_has_four_levels_on_three_tasks_plus_dense(self):
        configurations = levels_run()["configurations"]
        self.assertEqual(len(configurations), 4 * 3 + 3)
        # Longest task first, dense (already recorded) last.
        self.assertEqual([c["tasks"][0] for c in configurations[:12:4]], ["gsm8k", "humaneval_instruct", "ceval-valid"])
        self.assertTrue(all(c["name"].endswith("_dense") for c in configurations[12:]))
        split = [c for c in configurations if "out64_3b_2b" in c["name"]]
        self.assertEqual(len(split), 3)
        step = split[0]["kv"]["pipeline"][0]
        self.assertEqual((step["bits"], step["outliers"], step["signs"]), (2, 64, "rope_pairs"))
        self.assertEqual(split[0]["metadata"]["kv_compression"], 1 - 2.5 / 16)
        by_tag = {c["name"].rsplit("_turboquant_mse_", 1)[-1]: c["metadata"]["kv_compression"]
                  for c in configurations if "_turboquant_mse_all" in c["name"]}
        self.assertEqual(by_tag, {"all4": 1 - 4 / 16, "all3": 1 - 3 / 16, "all2": 1 - 2 / 16})


if __name__ == "__main__":
    unittest.main()
