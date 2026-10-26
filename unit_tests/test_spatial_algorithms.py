"""Spatial pair formats, per-head padded window budgets, query-aware keys, and decode."""
import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from compression_topics.spatial.algorithms import methods, windows
from compression_topics.spatial.algorithms.formats import (
    HEADER_BITS, RESIDUALS, TAG_BITS, WINDOW, dense_bits, kept_entries, max_saving, pair_bits)
from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress import metrics, parse_kv_spec


def correlated(length=256, heads=2, dim=128, noise=.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks = -(-length // 8)
    base = torch.randn(1, heads, blocks, 1, dim, generator=g).expand(1, heads, blocks, 8, dim)
    x = (base + noise * torch.randn(1, heads, blocks, 8, dim, generator=g)).reshape(1, heads, 8 * blocks, dim)
    return x[..., :length, :].contiguous()


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


BITS_OF = torch.tensor([dense_bits(2, 128)] + [pair_bits(r, 128) for r in RESIDUALS]) + TAG_BITS


def head_bits(code):
    """Bits each head uses per window [heads, windows], tags included."""
    return BITS_OF[code[..., ::2]].sum(-1)


class FormatTests(unittest.TestCase):
    def test_bits(self):
        self.assertEqual(pair_bits(0, 128), 2080)
        self.assertEqual(pair_bits(16, 128), 2464)
        self.assertEqual(pair_bits(32, 128), 2720)
        self.assertAlmostEqual(max_saving(128), 1 - 2082 / 4096)
        with self.assertRaises(ValueError):
            pair_bits(8, 128)

    def test_wide_heads_keep_a_quarter(self):
        self.assertEqual((kept_entries(32, 128), kept_entries(32, 256)), (32, 64))
        self.assertEqual(pair_bits(32, 256), 16 * (256 + 64 + 2) + 256)
        x = correlated(64, heads=4, dim=256).to(torch.bfloat16)
        out, bits, _ = windows.compress(x, target='k', saving=.3)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(bits, int(4 * 4 * windows.block_bits(256, .3)))


class WindowTests(unittest.TestCase):
    def test_every_head_meets_the_saving_in_every_window(self):
        x = correlated(512, heads=4, noise=.5)
        for saving in (.1, .3, .4):
            errors, _ = windows.candidates(x, target='v')
            code = windows.solve(errors, 128, saving)
            self.assertEqual(code.shape, (4, 512 // WINDOW, WINDOW))
            self.assertTrue(bool((head_bits(code) <= windows.block_bits(128, saving)).all()))

    def test_heads_do_not_share_budgets(self):
        x = correlated(256, heads=4, noise=.6)
        x[:, 0, 1::2] = x[:, 0, ::2]  # head 0 merges for free but cannot give bits away
        errors, _ = windows.candidates(x, target='v')
        code = windows.solve(errors, 128, .3)
        self.assertTrue(bool((head_bits(code) <= windows.block_bits(128, .3)).all()))
        self.assertTrue(bool((code[0] != 0).any()))

    def test_close_to_exhaustive_search(self):
        x = correlated(16, heads=1, noise=.6)
        errors, _ = windows.candidates(x, target='v')
        cost = lambda i, c: 0. if c == 0 else float(errors[0, i, c - 1])
        for saving in (.1, .25, .4):
            limit = (1 - saving) * dense_bits(WINDOW, 128) + 8 * TAG_BITS
            best = min(sum(cost(i, c) for i, c in enumerate(combo))
                       for combo in itertools.product(range(4), repeat=8)
                       if sum(BITS_OF[c] for c in combo) <= limit)
            code = windows.solve(errors, 128, saving)
            chosen = sum(cost(i, int(code[0, 0, 2 * i])) for i in range(8))
            self.assertLessEqual(chosen, best * 1.5 + 1e-6)

    def test_windows_are_independent(self):
        x = correlated(256, noise=.5)
        changed = x.clone()
        changed[..., 128:, :] *= -3
        code = lambda t: windows.solve(windows.candidates(t, target='v')[0], 128, .4)
        self.assertTrue(torch.equal(code(x)[:, :8], code(changed)[:, :8]))

    def test_unreachable_target_merges_every_pair(self):
        errors, _ = windows.candidates(correlated(64), target='v')
        code = windows.solve(errors, 128, .5)
        self.assertTrue(bool((code == 1).all()))  # PR0 everywhere

    def test_padded_blocks(self):
        x = correlated(128, heads=4, noise=.6)
        _, bits, _ = windows.compress(x, target='v', saving=.25)
        self.assertEqual(bits, int(4 * 8 * windows.block_bits(128, .25)))  # every block in full
        identical = x.clone()
        identical[:, :, 1::2] = identical[:, :, ::2]
        out, bits, counts = windows.compress(identical, target='v', saving=.5)
        self.assertEqual(counts, {"format_dense_pair": 0, "format_g2_r0": 4 * 64})
        self.assertEqual(bits, 4 * 8 * int(head_bits(torch.ones(1, 1, WINDOW, dtype=torch.long))))
        torch.testing.assert_close(out, identical, atol=3e-2, rtol=3e-2)  # identical pairs rebuild

    @unittest.skipUnless(torch.cuda.is_available() or torch.backends.mps.is_available(), "needs a GPU")
    def test_on_gpu(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        out, bits, _ = windows.compress(correlated(64).to(device), target='v', saving=.3)
        self.assertEqual(out.device.type, device)
        self.assertEqual(bits, int(2 * 4 * windows.block_bits(128, .3)))


class StreamingTests(unittest.TestCase):
    def tearDown(self):
        methods.reset()
        metrics.reset_stats()

    def test_prefill_then_decode_equals_one_pass(self):
        x = correlated(203, noise=.4)
        whole, whole_stats = cache_run(x, [203], saving=.35)
        streamed, stream_stats = cache_run(x, [21] + [1] * 182, saving=.35)
        torch.testing.assert_close(streamed, whole)
        for key in ('dense_bits', 'metadata_bits'):
            self.assertEqual(stream_stats[key], whole_stats[key])
        # Stored bits round down per update, so they may differ by under one bit per update.
        self.assertLessEqual(abs(stream_stats['stored_bits'] - whole_stats['stored_bits']), 12)
        self.assertTrue(torch.equal(streamed[..., 192:, :], x[..., 192:, :]))  # open window

    def test_first_window_is_compressed_when_complete(self):
        x = correlated(32)
        cache, stats = cache_run(x, [15], saving=.3)
        self.assertTrue(torch.equal(cache, x[..., :15, :]))  # not yet a whole window
        cache, stats = cache_run(x, [16], saving=.3)
        self.assertFalse(torch.equal(cache, x[..., :16, :]))
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
        step = SpatialConfig().pipeline(.3, k_layers=[1], v_layers=[])
        self.assertEqual(step['pipeline'][0], dict(method='spatial', saving=.3, k_layers=[1], v_layers=[]))
        self.assertEqual(parse_kv_spec(step).pipeline[0].method, 'spatial')
        with self.assertRaises(ValueError):
            SpatialConfig(query_weights='q.pt')  # no checksum
        with self.assertRaises(ValueError):
            parse_kv_spec({'pipeline': [dict(SpatialConfig().pipeline(.3)['pipeline'][0], typo=1)]})


class QueryAwareTests(unittest.TestCase):
    def test_query_weights_change_keys_only(self):
        x = correlated(256, noise=.6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'q.pt'
            weights = torch.ones(1, 2, 128)
            weights[..., :8] = 50.
            weights[..., 64:72] = 50.
            torch.save({'weights': weights}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            aware = dict(query_weights=str(path), query_weights_sha256=digest)

            def run(target, **options):
                methods.reset()
                cache = x.clone()
                methods.after_append(cache, target=target, layer_idx=0, start=0, end=256,
                                     saving=.4, **options)
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
        self.assertEqual(row['stored_bits'], HEADER_BITS + 4 * int(windows.block_bits(64, .4))
                         + dense_bits(6, 64))


if __name__ == '__main__':
    unittest.main()
