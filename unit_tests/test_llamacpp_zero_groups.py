"""CPU tests for counting zero-groups in llama.cpp's key layout."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import vector_compress, vector_compress_pair
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.spec import parse_kv_spec

import compression_topics.vector.scripts.llamacpp_zero_groups as study


def gguf_permute(weights: torch.Tensor, n_head: int) -> torch.Tensor:
    """llama.cpp's LlamaModel.permute (conversion/llama.py), applied to q/k weights."""
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )


def rope_half_split(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """transformers' rotate_half RoPE: pairs (i, i + d/2)."""
    half = x.shape[-1] // 2
    cos, sin = theta.cos(), theta.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def rope_adjacent(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """llama.cpp's NORM RoPE: pairs (2i, 2i + 1)."""
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = theta.cos(), theta.sin()
    return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)


class LayoutTests(unittest.TestCase):
    def test_reorder_matches_gguf_weight_permutation(self):
        torch.manual_seed(0)
        n_kv_heads, head_dim, hidden = 2, 8, 12
        weight = torch.randn(n_kv_heads * head_dim, hidden)
        x = torch.randn(5, hidden)
        hf_keys = (x @ weight.T).reshape(5, n_kv_heads, head_dim)
        cpp_keys = (x @ gguf_permute(weight, n_kv_heads).T).reshape(5, n_kv_heads, head_dim)
        self.assertTrue(torch.allclose(study.to_llamacpp_order(hf_keys), cpp_keys))

    def test_rope_commutes_with_reorder(self):
        torch.manual_seed(1)
        x = torch.randn(3, 16)
        theta = torch.rand(8) * 6
        rotated_then_reordered = study.to_llamacpp_order(rope_half_split(x, theta))
        reordered_then_rotated = rope_adjacent(study.to_llamacpp_order(x), theta)
        self.assertTrue(torch.allclose(rotated_then_reordered, reordered_then_rotated, atol=1e-6))

    def test_paired_zeros_are_adjacent_in_llamacpp_order(self):
        torch.manual_seed(2)
        keys = torch.randn(1, 2, 3, 16)
        spec = parse_kv_spec(vector_compress_pair(k_layers="all", v_layers=[], prune_pct=50))
        out, _ = compress_kv(keys, keys.clone(), layer_idx=0, spec=spec)
        cpp = study.to_llamacpp_order(out).reshape(1, 2, 3, 8, 2)
        zero = cpp == 0
        self.assertTrue(torch.equal(zero[..., 0], zero[..., 1]))
        self.assertTrue(torch.equal(zero[..., 0].sum(-1), torch.full((1, 2, 3), 4)))


class CountingTests(unittest.TestCase):
    def test_zero_group_starts(self):
        x = torch.tensor([[1.0, 0, 0, 2, 0, 3, 0, 0]])
        self.assertEqual(int(study.zero_group_starts(x).sum()), 3)

    def test_profiler_counts_keys_in_llamacpp_order(self):
        # transformers layout [a0 a1 a2 a3 | b0 b1 b2 b3]; zero a1 and b1 ->
        # llama.cpp layout a0 b0 a1 b1 a2 b2 a3 b3 has one group of length 2.
        keys = torch.tensor([[1.0, 0, 1, 1, 1, 0, 1, 1]])
        values = torch.tensor([[0.0, 1, 0, 1, 1, 1, 1, 1]])
        profiler = study.Profiler(count_values=True)
        profiler.record(keys, values)
        k, v = profiler.k.summary(), profiler.v.summary()
        self.assertEqual(k["groups_per_vector"], 1)
        self.assertEqual(k["avg_group_length"], 2)
        self.assertEqual(v["groups_per_vector"], 2)
        self.assertEqual(profiler.k_pairs.summary()["groups_per_vector"], 1)

    def test_same_values_zeroed_in_either_layout(self):
        torch.manual_seed(3)
        keys = torch.randn(1, 1, 4, 128)
        spec = parse_kv_spec(vector_compress(prune_pct=30))
        out, _ = compress_kv(keys, keys.clone(), layer_idx=0, spec=spec)
        cpp = study.to_llamacpp_order(out)
        self.assertEqual(int((out == 0).sum()), int((cpp == 0).sum()))
        self.assertEqual(int((out == 0).sum()), 4 * 38)


class RowTests(unittest.TestCase):
    def test_rle_saving_matches_deck_formula(self):
        self.assertAlmostEqual(study.rle_net_saving(64, 32.04), (64 - 32.04) / 128)
        self.assertAlmostEqual(study.random_groups(128, 64), 32.5)
        self.assertAlmostEqual(study.random_groups(64, 32), 16.5)

    def test_configurations(self):
        scalar = study.configurations("scalar")
        pair = study.configurations("pair")
        self.assertEqual([c["kv"]["pipeline"][0]["prune_pct"] for c in scalar], list(study.PERCENTAGES))
        self.assertEqual(scalar[0]["kv"]["pipeline"][0]["method"], "vector_compress")
        self.assertEqual(pair[0]["kv"]["pipeline"][0]["v_layers"], [])
        self.assertEqual(scalar[0]["limit"], 1)
        self.assertEqual(scalar[0]["tasks"], ["ceval-valid"])


if __name__ == "__main__":
    unittest.main()
