"""Fixed-ratio chunks: every eight-token window meets the saving on its own, in prefill and decode."""
import itertools
import unittest

import torch

from compression_topics.spatial.algorithms import chunked, methods, windows
from compression_topics.spatial.algorithms.formats import HEADER_BITS, TAG_BITS, WINDOW, dense_bits, max_saving
from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress import metrics, parse_kv_spec
from unit_tests.test_spatial_algorithms import LiveModelTests, correlated


def per_window(x, saving, **options):
    tables, payload, _ = windows.candidates(x, target='v', **options)
    return chunked.solve(tables, payload, dense_bits(WINDOW, x.shape[-1]), x.shape[1], saving)


def run(x, sizes, **step):
    methods.reset()
    metrics.reset_stats()
    cache, start = x[..., :0, :].clone(), 0
    for size in sizes:
        cache = torch.cat((cache, x[..., start:start + size, :]), -2)
        methods.chunked_after_append(cache, target='v', layer_idx=0, start=start, end=start + size, **step)
        start += size
    return cache, dict(metrics.STATS['v_layer_0'])


class ChunkTests(unittest.TestCase):
    def tearDown(self):
        methods.reset()
        metrics.reset_stats()

    def test_every_window_meets_the_saving(self):
        x = correlated(512, heads=4, noise=.5)
        for saving, options in ((.3, {}), (.4, {}), (.6, {}), (.3, dict(max_group=2))):
            bits, _ = per_window(x, saving, **options)
            limit = (1 - saving) * 4 * dense_bits(WINDOW, 128)
            self.assertTrue(bool((bits <= limit).all()))
            self.assertGreater(1 - bits.sum() / (len(bits) * 4 * dense_bits(WINDOW, 128)), saving)

    def test_window_choice_matches_exhaustive_search_when_tight(self):
        x = correlated(8, heads=2, noise=.6)
        tables, payload, _ = windows.candidates(x, target='v', max_group=2)
        menu = [(0., payload[(2, None)])] + [(None, payload[(2, r)]) for r in (0, 16, 32)]
        losses = tables[2]  # [heads, pairs, 3]
        for saving in (.1, .25, .4):
            limit = (1 - saving) * 2 * dense_bits(WINDOW, 128)
            best = min(
                (sum(0. if c == 0 else float(losses[i // 4, i % 4, c - 1]) for i, c in enumerate(combo)), combo)
                for combo in itertools.product(range(4), repeat=8)
                if sum(menu[c][1] + TAG_BITS for c in combo) <= limit)
            bits, code = chunked.solve(tables, payload, dense_bits(WINDOW, 128), 2, saving)
            self.assertLessEqual(float(bits[0]), limit)
            chosen = sum(0. if code[i // 4, 0, 2 * (i % 4)] == 0 else
                         float(losses[i // 4, i % 4, int(code[i // 4, 0, 2 * (i % 4)]) - 1]) for i in range(8))
            self.assertLessEqual(chosen, best[0] * 1.5 + 1e-6)  # multiplier solution, near the exact optimum

    def test_windows_are_independent(self):
        x = correlated(256, noise=.5)
        changed = x.clone()
        changed[..., 128:, :] *= -3
        _, a = per_window(x, .4)
        _, b = per_window(changed, .4)
        self.assertTrue(torch.equal(a[:, :16], b[:, :16]))

    def test_prefill_then_decode_equals_one_pass(self):
        x = correlated(203, noise=.4)
        whole, whole_stats = run(x, [203], saving=.35)
        streamed, stream_stats = run(x, [21] + [1] * 182, saving=.35)
        torch.testing.assert_close(streamed, whole)
        for key in ('dense_bits', 'stored_bits', 'metadata_bits'):
            self.assertEqual(stream_stats[key], whole_stats[key])
        self.assertTrue(torch.equal(streamed[..., 200:, :], x[..., 200:, :]))
        self.assertFalse(torch.equal(streamed[..., :8, :], x[..., :8, :]))  # no startup delay
        dense = whole_stats['dense_bits'] - 2 * dense_bits(3, 128)
        self.assertGreaterEqual(1 - (whole_stats['stored_bits'] - HEADER_BITS - 2 * dense_bits(3, 128)) / dense, .35)

    def test_infeasible_pair_target_takes_smallest_formats(self):
        x = correlated(64)
        bits, code = per_window(x, .5, max_group=2)
        self.assertTrue(bool((code > 0).all()))
        saving = 1 - bits.sum() / (len(bits) * 2 * dense_bits(WINDOW, 128))
        self.assertAlmostEqual(float(saving), max_saving(128), places=6)

    def test_config(self):
        step = SpatialConfig(algorithm='chunked', query_aware=False).pipeline(.3, k_layers=[0], v_layers=[])
        self.assertEqual(parse_kv_spec(step).pipeline[0].method, 'spatial_chunked')
        with self.assertRaises(ValueError):
            SpatialConfig(algorithm='chunked', control={})


class LiveChunkTests(LiveModelTests):
    def test_chunked_compresses_generated_tokens(self):
        model = self.model()
        dense_keys, _ = self.generate(model, {'pipeline': []})
        keys, stats = self.generate(model, SpatialConfig(algorithm='chunked').pipeline(.4))
        self.assertFalse(torch.equal(keys[..., 48:64, :], dense_keys[..., 48:64, :]))
        torch.testing.assert_close(keys[..., 64:, :], dense_keys[..., 64:, :])
        row = stats['k_layer_0']
        compressed = row['dense_bits'] - dense_bits(6, 64)
        self.assertGreaterEqual(1 - (row['stored_bits'] - HEADER_BITS - dense_bits(6, 64)) / compressed, .4)


if __name__ == '__main__':
    unittest.main()
