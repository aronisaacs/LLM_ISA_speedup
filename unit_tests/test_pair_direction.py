"""Directional compression invariants, with no model load."""
import unittest
import torch
from compression_topics.spatial.algorithms.pair_rank import merge_top_pairs, _shift_rope
from compression_topics.spatial.scripts.pair_direction_profile import DirectionProfiler
from engine.kv_compress.rope import RopeTables


class DirectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.rope = RopeTables(10000., 128)
        self.keys = torch.randn(1, 2, 9, 128)
        self.keys[..., 1::2, :] *= 3

    def merge(self, keys, pct=100, keep=25):
        return merge_top_pairs(keys, pct=pct, keep_pct=keep, rope=True,
                               direction=True, rope_tables=self.rope)

    def test_norms_and_unselected_tokens_are_preserved(self):
        out = self.merge(self.keys)
        torch.testing.assert_close(out.norm(dim=-1), self.keys.norm(dim=-1))
        self.assertTrue(torch.equal(out[..., -1, :], self.keys[..., -1, :]))
        self.assertTrue(torch.equal(self.merge(self.keys, pct=0), self.keys))

    def test_full_residual_restores_keys(self):
        torch.testing.assert_close(self.merge(self.keys, keep=100), self.keys, atol=2e-6, rtol=2e-6)

    def test_same_direction_different_norms_is_exact(self):
        first = self.keys[..., :1, :]
        second = _shift_rope(3 * first, self.rope, inverse=False)
        keys = torch.cat((first, second), dim=-2)
        torch.testing.assert_close(self.merge(keys, keep=0), keys)

    def test_zero_and_antipodal_pairs_stay_exact(self):
        first = self.keys[..., :1, :]
        for second in (torch.zeros_like(first), _shift_rope(-first, self.rope, inverse=False)):
            keys = torch.cat((first, second), dim=-2)
            torch.testing.assert_close(self.merge(keys, keep=0), keys)

    def test_cosine_ranking_is_independent_of_token_scale(self):
        out = self.merge(self.keys, pct=50, keep=0)
        scale = torch.linspace(.5, 3., 9).reshape(1, 1, 9, 1)
        scaled = self.merge(self.keys * scale, pct=50, keep=0)
        torch.testing.assert_close(scaled, out * scale)

    def test_profile_counts_odd_sequence_and_residual_sizes(self):
        profiler = DirectionProfiler(self.rope)
        profiler.observe(0, self.keys, self.keys, 0)
        rows = profiler.payload()
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(row['pairs'], 4)
            self.assertEqual(sum(row['adjusted_cos_hist']), 4)
            self.assertEqual(sum(row['reconstruction_cos_hist']), 8)


if __name__ == '__main__':
    unittest.main()
