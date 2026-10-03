"""CPU tests for row-wide checksparse and its study driver."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from catalog.compressions import checksparse_row
from engine.eval_runner.progress import kv_brief
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.apply import kv_for_slot
from engine.layer_select.budgets import BUDGETS, LLAMA31_LAYERS, TASKS
from engine.layer_select.levels import LEVELS
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import ScoreRow
from engine.layer_select.slots import all_slots

import compression_topics.checksparse.scripts.checksparse_row_study as study


def _reference(tensor: torch.Tensor, pct: float, tile: int = 8) -> torch.Tensor:
    """Mustafa's _apply_tile_sparsity, without the config lookup."""
    bsz, kv_heads, seq_len, head_dim = tensor.shape
    width = kv_heads * head_dim
    flat = tensor.permute(0, 2, 1, 3).reshape(bsz * seq_len, width).clone()
    tiles = flat.view(flat.size(0), width // tile, tile)
    scores = tiles.abs().sum(dim=-1).to(torch.float32)
    prune_per_row = int(math.floor(scores.size(-1) * float(pct) / 100.0 + 1e-9))
    if prune_per_row == 0 and pct > 0.0:
        prune_per_row = 1
    prune_per_row = min(prune_per_row, scores.size(-1))
    _, indices = torch.topk(scores, prune_per_row, dim=-1, largest=False)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    tiles.masked_fill_(mask.unsqueeze(-1), 0)
    return flat.view(bsz, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _spec(prune_pct, k_layers="all", v_layers=()):
    return parse_kv_spec(checksparse_row(k_layers, list(v_layers) if v_layers != "all" else "all", prune_pct=prune_pct))


class ChecksparseRowTests(unittest.TestCase):
    def test_matches_mustafa_reference(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            key = torch.randn(2, 8, 5, 128, dtype=dtype)
            for pct in (1, 25, 50, 75, 100):
                out, _ = compress_kv(key, key, layer_idx=0, spec=_spec(pct))
                self.assertTrue(torch.equal(out, _reference(key, pct)), f"{dtype} {pct}%")

    def test_weak_head_gives_up_its_tiles(self):
        # Head 0 is strong, head 1 is weak. At 50% the row drops head 1 entirely.
        key = torch.cat((torch.ones(1, 1, 1, 16), torch.full((1, 1, 1, 16), 0.1)), dim=1)
        out, _ = compress_kv(key, key, layer_idx=0, spec=_spec(50))
        self.assertTrue(torch.equal(out[:, 0], key[:, 0]))
        self.assertTrue(torch.equal(out[:, 1], torch.zeros(1, 1, 16)))

    def test_targets_and_identity(self):
        key = torch.randn(1, 2, 3, 16)
        value = torch.randn(1, 2, 3, 16)
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=_spec(50))
        self.assertFalse(torch.equal(out_key, key))
        self.assertTrue(torch.equal(out_value, value))
        same_key, _ = compress_kv(key, value, layer_idx=0, spec=_spec(0))
        self.assertTrue(torch.equal(same_key, key))

    def test_fraction_removed_is_exact_at_rungs(self):
        key = torch.randn(1, 8, 4, 128) + 0.01
        for pct in LEVELS:
            out, _ = compress_kv(key, key, layer_idx=0, spec=_spec(pct))
            tiles = out.transpose(1, 2).reshape(1, 4, 128, 8)
            dropped = (tiles == 0).all(dim=-1).sum(dim=-1)
            self.assertTrue(torch.equal(dropped, torch.full((1, 4), 128 * pct // 100)))

    def test_rungs_and_brief(self):
        self.assertEqual([r.level for r in rungs_for("checksparse_row")], list(LEVELS))
        step = kv_for_slot(checksparse_row(), all_slots(1)[0], level=75)["pipeline"][0]
        self.assertEqual(step["prune_pct"], 75)
        self.assertEqual(kv_brief(parse_kv_spec(checksparse_row())), "checksparse row tile=8 prune=50%")


def _scored():
    rows = []
    for slot in all_slots(LLAMA31_LAYERS):
        for index, level in enumerate(LEVELS):
            delta = (0.1 if slot.target == "v" else 1.0) * (index + 1) * (1 + slot.layer / 100)
            rows.append(
                ScoreRow(
                    slot=slot,
                    level=level,
                    ppl=10.0 + delta,
                    delta=delta,
                    path="",
                    kv=kv_for_slot(checksparse_row(prune_pct=25), slot, level=level),
                )
            )
    return 10.0, rows


class StudyTests(unittest.TestCase):
    def test_sweep_is_dense_then_every_slot_at_each_rung(self):
        configurations = study.sweep_run()["configurations"]
        self.assertEqual(len(configurations), 1 + LLAMA31_LAYERS * 2 * len(LEVELS))
        self.assertEqual(configurations[0]["kv"], {"pipeline": []})
        methods = {c["kv"]["pipeline"][0]["method"] for c in configurations[1:]}
        self.assertEqual(methods, {"checksparse_row"})
        self.assertEqual(configurations[1]["name"], "llama31_checksparse_row_k00_p25")
        self.assertTrue(configurations[1]["output_path"].startswith("compression_topics/checksparse/figures/row/"))
        self.assertEqual(configurations[1]["tasks"], ["wikitext"])

    def test_tasks_run_covers_every_budget_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with mock.patch.object(study, "FIGURES", folder), mock.patch.object(
                study, "load_sweep_scores", return_value=_scored()
            ), mock.patch.object(study, "_rel", side_effect=str):
                path = study.write_tasks_run(folder / "budgets_run.json")
            run = json.loads(path.read_text())
            self.assertTrue((folder / "selected_p60.json").is_file())

        configurations = run["configurations"]
        self.assertEqual(len(configurations), len(TASKS) * (1 + len(BUDGETS)))
        compressed = [c for c in configurations if c["kv"]["pipeline"]]
        self.assertEqual(
            {step["method"] for c in compressed for step in c["kv"]["pipeline"]}, {"checksparse_row"}
        )
        self.assertIn("llama31_gsm8k_checksparse_row_p10", [c["name"] for c in configurations])
        budgets = sorted({c["metadata"]["kv_budget"] for c in compressed})
        self.assertEqual(tuple(budgets), BUDGETS)

    def test_command_flags(self):
        full = study._multi_run_command(Path("run.json"), Path("/tmp/out"), True)
        self.assertIn("--skip-existing", full)
        self.assertEqual(full[-3:], ["--results-root", "/tmp/out", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
