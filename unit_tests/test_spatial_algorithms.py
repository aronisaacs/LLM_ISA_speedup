"""Spatial formats, per-head tree choices, the online controller across decode, and offline."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from compression_topics.spatial.algorithms import methods, windows
from compression_topics.spatial.algorithms.controller import Control, Online, offline_price
from compression_topics.spatial.algorithms.formats import HEADER_BITS, TAG_BITS, group_bits, max_saving
from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress import metrics, parse_kv_spec
from engine.kv_compress.rope import RopeTables


def correlated(length=256, heads=2, dim=128, noise=.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks = -(-length // 8)
    base = torch.randn(1, heads, blocks, 1, dim, generator=g).expand(1, heads, blocks, 8, dim)
    x = (base + noise * torch.randn(1, heads, blocks, 8, dim, generator=g)).reshape(1, heads, 8 * blocks, dim)
    return x[..., :length, :].contiguous()


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
    def test_heads_choose_independently_at_one_price(self):
        x = torch.randn(1, 2, 64, 128)
        x[:, 0, 1::2] = x[:, 0, ::2]  # head 0: identical neighbours
        table, _ = windows.build(x, target='v', max_group=2)
        choice = table.choose(0, .01)
        self.assertEqual(choice.heads[0].segments[0], (0, 2, 0))
        self.assertEqual(choice.heads[1].segments[0], (0, 2, None))

    def test_tree_prefers_larger_groups_when_tokens_agree(self):
        x = correlated(64, noise=0.)
        table, recon = windows.build(x, target='v', max_group=8)
        choice = table.choose(0, 1.)
        self.assertEqual(choice.heads[0].segments, ((0, 8, 0),))
        out = windows.restore(x, [table.choose(w, 1.) for w in range(len(table))], recon)
        torch.testing.assert_close(out, x, atol=2e-2, rtol=2e-2)

    def test_fixed_residual_restricts_pairs(self):
        x = correlated(128)
        table, _ = windows.build(x, target='v', max_group=2, fixed_residual=16)
        for w in range(len(table)):
            for price in (0., .1, 1e6):
                for head in table.choose(w, price).heads:
                    self.assertTrue(all(size == 2 and r in (None, 16) for _, size, r in head.segments))
        with self.assertRaises(ValueError):
            windows.build(x, target='v', max_group=8, fixed_residual=16)


class ControllerTests(unittest.TestCase):
    def run_online(self, x, saving, **options):
        table, _ = windows.build(x, target='v', **options)
        c = Online(saving, Control())
        c.start(table)
        return c, [c.decide(table, w)[0] for w in range(len(table))]

    def test_online_reaches_target_and_is_causal(self):
        x = correlated(2048, noise=.5)
        c, choices = self.run_online(x, .4)
        self.assertLess(abs((1 - c.stored / c.dense) - .4), .03)
        changed = x.clone()
        changed[:, :, 1024:] *= 3
        _, again = self.run_online(changed, .4)
        self.assertEqual([ch.heads for ch in choices[:128]], [ch.heads for ch in again[:128]])

    def test_minimum_storage_merges_every_pair(self):
        x = correlated(128)
        table, _ = windows.build(x, target='v', max_group=2)
        c = Online(.5, Control(), minimum_storage=True)
        c.start(table)
        choices = [c.decide(table, w)[0] for w in range(len(table))]
        self.assertTrue(all(s[1:] == (2, 0) for ch in choices for h in ch.heads for s in h.segments))
        pairs = 2 * 64
        self.assertEqual(c.stored, pairs * (group_bits(0, 128, 2) + TAG_BITS) + HEADER_BITS)

    def test_offline_meets_budget(self):
        x = correlated(512, noise=.5)
        table, _ = windows.build(x, target='v', max_group=8)
        price = offline_price(table, .5)
        stored = sum(table.choose(w, price).bits for w in range(len(table))) + HEADER_BITS
        saving = 1 - stored / (len(table) * table.window_bits)
        self.assertGreaterEqual(saving, .5)
        self.assertLess(saving, .52)


def cache_run(x, chunks, step):
    """Feed ``x`` through ``online_after_append`` in ``chunks`` (a growing cache tensor)."""
    methods.reset()
    metrics.reset_stats()
    cache = x[..., :0, :].clone()
    start = 0
    for size in chunks:
        cache = torch.cat((cache, x[..., start:start + size, :]), -2)
        methods.online_after_append(cache, target='v', layer_idx=0, start=start, end=start + size, **step)
        start += size
    return cache, dict(metrics.STATS['v_layer_0'])


class StreamingTests(unittest.TestCase):
    def tearDown(self):
        methods.reset()
        metrics.reset_stats()

    def test_prefill_then_decode_equals_one_pass(self):
        x = correlated(203, noise=.4)
        step = dict(saving=.3, max_group=8)
        whole, whole_stats = cache_run(x, [203], step)
        streamed, stream_stats = cache_run(x, [21] + [1] * 182, step)  # short prompt, then decode
        torch.testing.assert_close(streamed, whole)
        for key in ('dense_bits', 'stored_bits', 'metadata_bits'):
            self.assertEqual(stream_stats[key], whole_stats[key])
        self.assertTrue(torch.equal(streamed[..., 200:, :], x[..., 200:, :]))  # unfinished window stays dense
        self.assertFalse(torch.equal(streamed[..., :8, :], x[..., :8, :]))

    def test_tokens_wait_for_startup(self):
        x = correlated(40)
        cache, _ = cache_run(x, [24], dict(saving=.3))
        self.assertTrue(torch.equal(cache, x[..., :24, :]))  # fewer than 32 tokens: still dense
        cache, stats = cache_run(x, [24] + [1] * 16, dict(saving=.3))
        self.assertFalse(torch.equal(cache[..., :32, :], x[..., :32, :]))
        self.assertGreater(stats['dense_bits'], stats['stored_bits'])

    def test_new_sequence_resets_state(self):
        x = correlated(64)
        first, _ = cache_run(x, [64], dict(saving=.3))
        methods.online_after_append(x.clone(), target='v', layer_idx=0, start=0, end=64, saving=.3)
        again = x.clone()
        methods.online_after_append(again, target='v', layer_idx=0, start=0, end=64, saving=.3)
        torch.testing.assert_close(again, first)


class ConfigTests(unittest.TestCase):
    def test_pipeline_and_validation(self):
        step = SpatialConfig(algorithm='offline', max_group=4).pipeline(.3, k_layers=[1], v_layers=[])
        spec = parse_kv_spec(step)
        self.assertEqual(spec.pipeline[0].method, 'spatial_offline')
        for options in (dict(max_group=3), dict(query_aware=True), dict(algorithm='offline', control={}),
                        dict(max_group=8, fixed_residual=16), dict(algorithm='offline', minimum_storage=True),
                        dict(max_group=8, minimum_storage=True)):
            with self.assertRaises(ValueError):
                SpatialConfig(**options)
        with self.assertRaises(ValueError):
            parse_kv_spec({'pipeline': [dict(SpatialConfig().pipeline(.3)['pipeline'][0], typo=1)]})


class QueryAwareTests(unittest.TestCase):
    def test_query_weights_change_key_choices_only(self):
        torch.manual_seed(1)
        rope = RopeTables(rope_theta=10000., head_dim=128)
        x = correlated(256, noise=.6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'q.pt'
            weights = torch.ones(1, 2, 128)
            weights[..., :8] = 50.
            weights[..., 64:72] = 50.
            torch.save({'weights': weights}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            options = dict(layer_idx=0, saving=.4, rope_tables=rope)
            plain = methods.offline(x, target='k', **options)
            aware = methods.offline(x, target='k', query_aware=True, query_weights=str(path),
                                    query_weights_sha256=digest, **options)
            self.assertFalse(torch.equal(plain, aware))
            values = methods.offline(x, target='v', query_aware=True, query_weights=str(path),
                                     query_weights_sha256=digest, **options)
            torch.testing.assert_close(values, methods.offline(x, target='v', **options))
            with self.assertRaises(ValueError):
                methods.offline(x, target='k', query_aware=True, query_weights=str(path),
                                query_weights_sha256='0' * 64, **options)
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
                out = model(input_ids=ids, past_key_values=cache, use_cache=True)
                for position in range(prompt, prompt + new):  # fixed tokens: layer-0 keys depend only on ids
                    token = torch.tensor([[(7 * position) % 32]])
                    out = model(input_ids=token, past_key_values=cache, use_cache=True)
            return cache.layers[0].keys.clone(), metrics.pop_stats()
        finally:
            uninstall()

    def test_online_compresses_generated_tokens_offline_does_not(self):
        model = self.model()
        dense_keys, _ = self.generate(model, {'pipeline': []})
        online_keys, online = self.generate(model, SpatialConfig(max_group=8).pipeline(.4))
        offline_keys, offline = self.generate(model, SpatialConfig(algorithm='offline').pipeline(.4))
        self.assertEqual(online_keys.shape[-2], 70)
        decoded = slice(48, 64)  # windows written entirely during generation
        self.assertFalse(torch.equal(online_keys[..., decoded, :], dense_keys[..., decoded, :]))
        torch.testing.assert_close(offline_keys[..., decoded, :], dense_keys[..., decoded, :])
        torch.testing.assert_close(online_keys[..., 64:, :], dense_keys[..., 64:, :])  # open window
        for stats in (online, offline):
            self.assertEqual(stats['k_layer_0']['dense_bits'], 70 * 64 * 16)
        self.assertGreater(online['k_layer_0']['dense_bits'] - online['k_layer_0']['stored_bits'],
                           offline['k_layer_0']['dense_bits'] - offline['k_layer_0']['stored_bits'])


if __name__ == '__main__':
    unittest.main()
