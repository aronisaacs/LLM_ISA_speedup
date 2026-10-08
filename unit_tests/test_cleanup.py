"""Regression checks for shared execution, result integrity, and byte accounting."""
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from engine.eval_runner.index import record_simulation, find_result, rebuild
from engine.eval_runner.cache import simulation_identity, identities_match
from engine.kv_compress.spec import parse_kv_spec
from engine.kv_compress import metrics


def writer(root, index, ready):
    ready.wait()
    record_simulation({'pretrained': str(index), 'kv': {'pipeline': []}},
                      {'task': {'score': index}}, root=Path(root))


class LedgerTests(unittest.TestCase):
    def test_parallel_writers_preserve_every_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = multiprocessing.get_context('fork')
            ready = context.Event()
            processes = [context.Process(target=writer, args=(tmp, i, ready)) for i in range(12)]
            for process in processes:
                process.start()
            ready.set()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            rows = json.loads((Path(tmp) / 'results.json').read_text())['simulations']
            self.assertEqual({r['identity']['pretrained'] for r in rows}, {str(i) for i in range(12)})

    def test_corrupt_ledger_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.json'
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                record_simulation({}, {}, root=Path(tmp))
            self.assertEqual(path.read_text(), '{broken')

    def test_rebuild_preserves_ledger_only_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = record_simulation({'pretrained': 'saved'}, {'task': {'score': .4}}, root=Path(tmp))
            self.assertEqual(rebuild(Path(tmp)), [row])

    def test_prompt_model_and_seed_options_change_identity(self):
        base = {'model_args': 'pretrained=test/model,dtype=float16', 'tasks': ['test']}
        kv = parse_kv_spec(None)
        dense = simulation_identity(base, {}, kv)
        for override in ({'apply_chat_template': True}, {'fewshot_random_seed': 4},
                         {'system_instruction': 'Answer briefly'},
                         {'model_args': 'pretrained=test/model,dtype=float16,revision=other'}):
            self.assertFalse(identities_match(dense, simulation_identity(base, override, kv)))
        legacy = {k: v for k, v in dense.items()
                  if k not in ('model', 'model_options', 'evaluation_options', 'apply_chat_template')}
        self.assertTrue(identities_match(legacy, dense))
        self.assertFalse(identities_match(legacy, {**dense, 'apply_chat_template': True}))

    def test_imported_result_keeps_its_chat_template_flag(self):
        from engine.eval_runner.cache import _legacy_from_payload
        payload = {'config': {'model_args': {'pretrained': 'm', 'dtype': 'float16',
                                              'kv': {'pipeline': []}},
                              'apply_chat_template': True},
                   'results': {'task': {'score': .5}}}
        imported = _legacy_from_payload(payload)
        self.assertTrue(imported['apply_chat_template'])
        plain = simulation_identity({'model_args': 'pretrained=m,dtype=float16',
                                     'tasks': ['task']}, {}, parse_kv_spec(None))
        self.assertFalse(identities_match(imported, plain))

    def test_planned_and_measured_savings_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = {'targets': {'k': {'compression': .2}, 'kv': {'compression': .1}}}
            row = record_simulation({}, {}, root=Path(tmp), budget=.25,
                planned_compression=.26, storage=storage, compression_target='k')
            self.assertEqual(row['planned_compression'], .26)
            self.assertEqual(row['measured_compression'], .2)
            self.assertNotIn('compression', row)


class InterfaceTests(unittest.TestCase):
    def test_unknown_option_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'unknown options'):
            parse_kv_spec({'pipeline': [{'method': 'group_calibrated', 'saving': .2,
                                        'resdual_entries': 8, 'k_layers': [0]}]})

    def test_measured_bytes_include_dense_layers_of_different_sizes(self):
        metrics.reset_stats()
        spec = parse_kv_spec({'pipeline': [{'method': 'group_calibrated', 'saving': .2, 'k_layers': [0]}]})
        small = torch.zeros(1, 1, 4, 8)
        large = torch.zeros(1, 2, 4, 8)
        metrics.observe(small, small, 0, spec)
        metrics.observe(large, large, 1, spec)
        metrics.STATS['k_layer_0'] = {'dense_bits': 512, 'stored_bits': 256}
        result = metrics.storage_summary()['targets']
        self.assertEqual(result['k']['dense_bits'], 1536)
        self.assertAlmostEqual(result['k']['compression'], 1/6)
        self.assertAlmostEqual(result['kv']['compression'], 1/12)
        metrics.reset_stats()

    def test_chunk_identity_uses_configured_model(self):
        from engine.eval_runner.chunks import identity
        config = {'kv': {'pipeline': []}}
        a = identity(config, {'seed': 0}, {'model_args': 'pretrained=first,dtype=float16'})
        b = identity(config, {'seed': 0}, {'model_args': 'pretrained=second,dtype=float16'})
        self.assertEqual(a['pretrained'], 'first')
        self.assertNotEqual(a, b)

    def test_group_and_pair_plans_accept_other_dimensions(self):
        from compression_topics.spatial.scripts import group_calibrated_study as group
        from compression_topics.spatial.scripts import pair_calibrated_study as pair
        for study in (group, pair):
            result = study.plan(Path('/tmp/plan'), layers=2, head_dim=64,
                                model_args='pretrained=other,dtype=float16')
            self.assertEqual(result['head_dim'], 64)
            self.assertEqual(result['run']['model_args'], 'pretrained=other,dtype=float16')
            self.assertEqual({r['layer'] for r in result['candidates']}, {0, 1})

class ExtensionTests(unittest.TestCase):
    def test_new_method_supplies_its_own_append_callback(self):
        from engine.kv_compress.methods import METHODS, AFTER_APPEND
        from engine.kv_compress.cache import patch_cache_update
        from transformers.cache_utils import DynamicCache
        calls = []
        def apply(tensor, *, layer_idx, target, amount=1, **context):
            return tensor
        def after(tensor, *, target, layer_idx, start, end, amount=1, **context):
            calls.append((target, layer_idx, start, end))
            tensor[..., start:end, :].add_(amount)
        with patch.dict(METHODS, {'test_callback': apply}), patch.dict(AFTER_APPEND, {'test_callback': after}):
            spec = parse_kv_spec({'pipeline': [{'method': 'test_callback', 'k_layers': [0], 'amount': 3}]})
            uninstall = patch_cache_update(spec)
            try:
                cache = DynamicCache()
                for _ in range(2):
                    keys, values = cache.update(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4), 0)
                self.assertTrue(torch.all(keys == 3))
                self.assertTrue(torch.all(values == 0))
                self.assertEqual(calls, [('k', 0, 0, 1), ('k', 0, 1, 2)])
            finally:
                uninstall()
                metrics.reset_stats()

    def test_historical_chunk_identity_is_reusable_without_model_loading(self):
        from engine.eval_runner.chunks import identity
        config = {'kv': {'pipeline': []}, 'num_fewshot': 0}
        base = {'model_args': 'pretrained=old/model,dtype=bfloat16'}
        sampling = {'seed': 0, 'chunks': 16}
        old = {'pretrained': 'old/model', 'dtype': 'bfloat16', 'tasks': ['wikitext_chunks'],
               'sampling': sampling, 'kv': config['kv']}
        self.assertTrue(identities_match(old, identity(config, sampling, base)))
        other = {'model_args': 'pretrained=new/model,dtype=bfloat16'}
        self.assertFalse(identities_match(old, identity(config, sampling, other)))
