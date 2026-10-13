"""Spatial formats, per-window budgets, query-aware keys, and compression across decode."""
import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from compression_topics.spatial.algorithms import methods, windows
from compression_topics.spatial.algorithms.formats import (
    HEADER_BITS, TAG_BITS, WINDOW, dense_bits, group_bits, max_saving)
from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress import metrics, parse_kv_spec
from engine.kv_compress.rope import RopeTables


def correlated(length=256, heads=2, dim=128, noise=.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks = -(-length // 8)
    base = torch.randn(1, heads, blocks, 1, dim, generator=g).expand(1, heads, blocks, 8, dim)
    x = (base + noise * torch.randn(1, heads, blocks, 8, dim, generator=g)).reshape(1, heads, 8 * blocks, dim)
    return x[..., :length, :].contiguous()


def per_window(x, saving, **options):
    losses, _ = windows.candidates(x, target='v', **options)
    return windows.solve(losses, x.shape[-1], saving)


def cache_run(x, sizes, **step):
    """Feed ``x`` through the cache hook in updates of ``sizes`` tokens (a growing cache)."""
    methods.reset()
    metrics.reset_stats()
    cache, start = x[..., :0, :].clone(), 0
    for size in sizes:
        cache = torch.cat((cache, x[..., start:start + size, :]), -2)
        methods.after_append(cache, target='v', layer_idx=0, start=start, end=start + size, **step)
        start += size
    return cache, dict(metrics.STATS['v_layer_0'])


class FormatTests(unittest.TestCase):
    def test_bits(self):
        self.assertEqual(group_bits(0, 128, 2), 2080)
        self.assertEqual(group_bits(32, 128, 2), 2720)
        self.assertEqual(group_bits(16, 128, 4), 3648)
        self.assertEqual(group_bits(16, 128, 8), 5248)
        self.assertAlmostEqual(max_saving(128), 1 - 2083 / 4096)
        with self.assertRaises(ValueError):
            group_bits(32, 128, 4)


class WindowTests(unittest.TestCase):
    def test_every_window_meets_the_saving(self):
        x = correlated(512, heads=4, noise=.5)
        for saving, options in ((.3, {}), (.4, {}), (.6, {}), (.3, dict(max_group=2))):
            bits, _ = per_window(x, saving, **options)
            limit = (1 - saving) * 4 * dense_bits(WINDOW, 128)
            self.assertTrue(bool((bits <= limit).all()))
            self.assertGreater(1 - bits.sum() / (len(bits) * 4 * dense_bits(WINDOW, 128)), saving)

    def test_heads_share_the_window_budget(self):
        x = correlated(64, noise=.6)
        x[:, 0, 1::2] = x[:, 0, ::2]  # head 0: identical neighbours, merging costs nothing
        _, code = per_window(x, .3, max_group=2)
        self.assertTrue(bool((code[0] == 1).all()))  # head 0 merges every pair, no residual
        self.assertGreater(int((code[1] != 1).sum()), 0)  # head 1 keeps residuals or dense pairs

    def test_close_to_exhaustive_search(self):
        x = correlated(8, heads=2, noise=.6)
        losses, _ = windows.candidates(x, target='v', max_group=2)
        bits = [dense_bits(2, 128)] + [group_bits(r, 128, 2) for r in (0, 16, 32)]
        loss = lambda i, c: 0. if c == 0 else float(losses[2][i // 4, i % 4, c - 1])
        for saving in (.1, .25, .4):
            limit = (1 - saving) * 2 * dense_bits(WINDOW, 128)
            best = min(sum(loss(i, c) for i, c in enumerate(combo))
                       for combo in itertools.product(range(4), repeat=8)
                       if sum(bits[c] + TAG_BITS for c in combo) <= limit)
            used, code = windows.solve(losses, 128, saving)
            self.assertLessEqual(float(used[0]), limit)
            chosen = sum(loss(i, int(code[i // 4, 0, 2 * (i % 4)])) for i in range(8))
            self.assertLessEqual(chosen, best * 1.5 + 1e-6)

    def test_pair_sort_matches_price_search(self):
        for seed, noise in ((0, .4), (1, .8), (2, 1.5)):
            x = correlated(256, heads=8, noise=noise, seed=seed) * torch.rand(1, 8, 1, 1, generator=torch.Generator().manual_seed(seed)) * 2
            losses, _ = windows.candidates(x, target='v', max_group=2)
            for saving in (.1, .3, .4, .5):
                bits, code = windows.solve_pairs(losses, 128, saving)
                old_bits, old_code = windows.solve(losses, 128, saving)
                self.assertTrue(torch.equal(code, old_code))
                self.assertTrue(torch.equal(bits, old_bits))

    def test_tree_merges_eights_when_tokens_agree(self):
        x = correlated(64, noise=0.)
        _, code = per_window(x, .5)
        eight_r0 = windows.options(8).index((8, 0)) + 1
        self.assertTrue(bool((code == eight_r0).any()))

    def test_windows_are_independent(self):
        x = correlated(256, noise=.5)
        changed = x.clone()
        changed[..., 128:, :] *= -3
        self.assertTrue(torch.equal(per_window(x, .4)[1][:, :16], per_window(changed, .4)[1][:, :16]))

    def test_unreachable_pair_target_merges_every_pair(self):
        bits, code = per_window(correlated(64), .5, max_group=2)
        self.assertTrue(bool((code == 1).all()))
        self.assertAlmostEqual(float(1 - bits.sum() / (len(bits) * 2 * dense_bits(WINDOW, 128))),
                               max_saving(128), places=6)


class StreamingTests(unittest.TestCase):
    def tearDown(self):
        methods.reset()
        metrics.reset_stats()

    def test_prefill_then_decode_equals_one_pass(self):
        x = correlated(203, noise=.4)
        whole, whole_stats = cache_run(x, [203], saving=.35)
        streamed, stream_stats = cache_run(x, [21] + [1] * 182, saving=.35)
        torch.testing.assert_close(streamed, whole)
        for key in ('dense_bits', 'stored_bits', 'metadata_bits'):
            self.assertEqual(stream_stats[key], whole_stats[key])
        self.assertTrue(torch.equal(streamed[..., 200:, :], x[..., 200:, :]))  # open window
        tail = 2 * dense_bits(3, 128)
        saving = 1 - (whole_stats['stored_bits'] - HEADER_BITS - tail) / (whole_stats['dense_bits'] - tail)
        self.assertGreaterEqual(saving, .35)

    def test_first_window_is_compressed_immediately(self):
        x = correlated(16)
        cache, stats = cache_run(x, [8], saving=.3)
        self.assertFalse(torch.equal(cache, x[..., :8, :]))
        self.assertEqual(stats['metadata_bits'], HEADER_BITS + TAG_BITS * sum(
            v for k, v in stats.items() if k.startswith('format_')))

    def test_new_sequence_restarts(self):
        x = correlated(64)
        first, _ = cache_run(x, [64], saving=.3)
        again = x.clone()
        methods.after_append(again, target='v', layer_idx=0, start=0, end=64, saving=.3)
        torch.testing.assert_close(again, first)


class ConfigTests(unittest.TestCase):
    def test_pipeline_and_validation(self):
        step = SpatialConfig(max_group=4).pipeline(.3, k_layers=[1], v_layers=[])
        self.assertEqual(parse_kv_spec(step).pipeline[0].method, 'spatial')
        for options in (dict(max_group=3), dict(query_aware=True)):
            with self.assertRaises(ValueError):
                SpatialConfig(**options)
        with self.assertRaises(ValueError):
            parse_kv_spec({'pipeline': [dict(SpatialConfig().pipeline(.3)['pipeline'][0], typo=1)]})


class QueryAwareTests(unittest.TestCase):
    def test_query_weights_change_keys_only(self):
        rope = RopeTables(rope_theta=10000., head_dim=128)
        x = correlated(256, noise=.6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'q.pt'
            weights = torch.ones(1, 2, 128)
            weights[..., :8] = 50.
            weights[..., 64:72] = 50.
            torch.save({'weights': weights}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            aware = dict(query_aware=True, query_weights=str(path), query_weights_sha256=digest)

            def run(target, **options):
                methods.reset()
                cache = x.clone()
                methods.after_append(cache, target=target, layer_idx=0, start=0, end=256,
                                     saving=.4, rope_tables=rope, **options)
                return cache

            self.assertFalse(torch.equal(run('k'), run('k', **aware)))
            torch.testing.assert_close(run('v'), run('v', **aware))
            with self.assertRaises(ValueError):
                run('k', **dict(aware, query_weights_sha256='0' * 64))
        methods.reset()
        metrics.reset_stats()


class LiveModelTests(unittest.TestCase):
    def model(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(5)
        return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=160,
                                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                            head_dim=64, max_position_embeddings=256)).eval()

    def generate(self, model, kv, prompt=40, new=30):
        from types import SimpleNamespace
        from transformers import DynamicCache
        from engine.kv_compress.install import install
        metrics.reset_stats()
        methods.reset()
        uninstall = install(SimpleNamespace(model=model), parse_kv_spec(kv))
        try:
            with torch.inference_mode():
                cache = DynamicCache()
                ids = torch.arange(prompt).remainder(32).unsqueeze(0)
                model(input_ids=ids, past_key_values=cache, use_cache=True)
                for position in range(prompt, prompt + new):  # fixed tokens: layer-0 keys depend only on ids
                    model(input_ids=torch.tensor([[(7 * position) % 32]]), past_key_values=cache, use_cache=True)
            return cache.layers[0].keys.clone(), metrics.pop_stats()
        finally:
            uninstall()

    def test_generated_windows_are_compressed(self):
        model = self.model()
        dense_keys, _ = self.generate(model, {'pipeline': []})
        keys, stats = self.generate(model, SpatialConfig().pipeline(.4))
        self.assertEqual(keys.shape[-2], 70)
        self.assertFalse(torch.equal(keys[..., 48:64, :], dense_keys[..., 48:64, :]))  # written in decode
        torch.testing.assert_close(keys[..., 64:, :], dense_keys[..., 64:, :])  # open window
        row = stats['k_layer_0']
        self.assertEqual(row['dense_bits'], 70 * 64 * 16)
        compressed = row['dense_bits'] - dense_bits(6, 64)
        self.assertGreaterEqual(1 - (row['stored_bits'] - HEADER_BITS - dense_bits(6, 64)) / compressed, .4)


if __name__ == '__main__':
    unittest.main()
