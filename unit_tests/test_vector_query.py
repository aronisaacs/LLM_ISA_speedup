"""CPU tests for query-aware vector compression and its net-budget arms."""

from __future__ import annotations

import unittest

import torch

from catalog.compressions import vector_compress, vector_compress_pair, vector_compress_query
from compression_topics.vector.algorithms.vector_compress_query import pair_query, scalar_query
from compression_topics.vector.bitmask import net_saving
from engine.kv_compress import parse_kv_spec
from engine.kv_compress.pipeline import compress_kv
from engine.layer_select.apply import kv_for_assignment, parse_singleton
from engine.layer_select.slots import Slot


def _kv(seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 8, 6, 128)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def _run(kv, keys, values, layer=3):
    return compress_kv(keys.clone(), values.clone(), layer, parse_kv_spec(kv))


class QueryRankingTests(unittest.TestCase):
    def test_flat_weights_match_plain_ranking(self):
        keys, _ = _kv()
        flat = torch.ones(1, 8, 1, 128)
        for level in (25, 50, 75):
            plain_scalar, _ = _run(vector_compress(k_layers=[3], v_layers=[], prune_pct=level), keys, keys)
            plain_pair, _ = _run(vector_compress_pair(k_layers=[3], v_layers=[], prune_pct=level), keys, keys)
            self.assertTrue(torch.equal(scalar_query(keys, flat, level), plain_scalar))
            self.assertTrue(torch.equal(pair_query(keys, flat, level), plain_pair))

    def test_weights_change_which_keys_are_zeroed(self):
        keys = torch.ones(1, 1, 1, 8)
        weights = torch.tensor([[[[1., 2, 3, 4, 5, 6, 7, 8]]]])
        self.assertEqual(scalar_query(keys, weights, 50)[0, 0, 0].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        # Planes (0,4) (1,5) (2,6) (3,7) weigh 6, 8, 10, 12.
        self.assertEqual(pair_query(keys, weights, 50)[0, 0, 0].tolist(), [0, 0, 1, 1, 0, 0, 1, 1])


class QueryMethodTests(unittest.TestCase):
    def test_keys_keep_the_level_and_pairs_stay_together(self):
        keys, values = _kv()
        for pairs in (False, True):
            for level in (25, 50, 75):
                kv = vector_compress_query(k_layers=[3], v_layers=[], prune_pct=level, pairs=pairs)
                out, _ = _run(kv, keys, values)
                self.assertAlmostEqual(float((out == 0).float().mean()), level / 100)
                if pairs:
                    self.assertTrue(torch.equal(out[..., :64] == 0, out[..., 64:] == 0))

    def test_values_match_plain_per_scalar(self):
        keys, values = _kv(1)
        for pairs in (False, True):
            _, query_values = _run(vector_compress_query(k_layers=[], v_layers=[3], prune_pct=50, pairs=pairs),
                                   keys, values)
            _, plain_values = _run(vector_compress(k_layers=[], v_layers=[3], prune_pct=50), keys, values)
            self.assertTrue(torch.equal(query_values, plain_values))

    def test_query_ranking_differs_from_plain_on_real_weights(self):
        keys, values = _kv(2)
        query, _ = _run(vector_compress_query(k_layers=[3], v_layers=[], prune_pct=50), keys, values)
        plain, _ = _run(vector_compress(k_layers=[3], v_layers=[], prune_pct=50), keys, values)
        self.assertFalse(torch.equal(query == 0, plain == 0))

    def test_rejects_missing_weights(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"pipeline": [{"method": "vector_compress_query", "k_layers": [0], "v_layers": [],
                                         "prune_pct": 50}]})

    def test_greedy_levels_and_bitmask(self):
        template = vector_compress_query(prune_pct=25, pairs=True)
        kv = kv_for_assignment(template, {Slot(0, "k"): 50, Slot(1, "v"): 75})
        self.assertEqual([step["prune_pct"] for step in kv["pipeline"]], [50, 75])
        self.assertEqual(parse_singleton(kv_for_assignment(template, {Slot(2, "k"): 25})), (Slot(2, "k"), 25))
        self.assertEqual(net_saving("vector_compress_pair_query", "k", 50), net_saving("vector_compress_pair", "k", 50))
        self.assertEqual(net_saving("vector_compress_query", "k", 50), net_saving("vector_compress", "k", 50))


if __name__ == "__main__":
    unittest.main()
