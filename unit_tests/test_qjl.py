"""Hadamard codebook rewrite and the WikiText run. No model load."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from engine.eval_runner.load_run import load_run
from engine.kv_compress.methods.qjl import apply, centroids, hadamard, sign_diagonal

ROOT = Path(__file__).resolve().parents[1]


class QjlRewriteTests(unittest.TestCase):
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


class QjlRunTests(unittest.TestCase):
    def test_run_is_three_post_rope_wikitext_configs(self):
        loaded = load_run(ROOT / "runs" / "llama31.py:qjl")
        names = [item["name"] for item in loaded["configurations"]]
        self.assertEqual(names, ["llama31_qjl_k", "llama31_qjl_v", "llama31_qjl_both"])
        step = loaded["configurations"][0]["kv"]["pipeline"][0]
        self.assertEqual(step["method"], "qjl")
        self.assertEqual(step["bits"], 4)
        self.assertNotIn("pre_rope", step)
        self.assertEqual(step["v_layers"], [])
        self.assertEqual(loaded["configurations"][2]["output_path"], "results/qjl/llama31_qjl_both.json")


if __name__ == "__main__":
    unittest.main()
