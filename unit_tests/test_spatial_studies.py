"""The staged ladder and fixed-rung studies: candidates, calibration, allocation, gates and tasks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from compression_topics.spatial.algorithms.formats import ACCOUNTING
from compression_topics.spatial.scripts import fixed_rungs, ladder, study
from engine.eval_runner.cache import simulation_identity
from engine.eval_runner.chunks import identity
from engine.eval_runner.files import write_json
from engine.kv_compress import parse_kv_spec


def fake_chunks(path, *, endpoint_saving=None):
    """Pretend each slot met its target; optionally pin minimum-storage runs."""
    run = json.loads(Path(path).read_text())
    sampling = run['sampling']
    for config in run['configurations']:
        steps = config['kv']['pipeline']
        stats, saving = {}, 0.
        for step in steps:
            achieved = endpoint_saving if step.get('minimum_storage') and endpoint_saving else step['saving']
            for target in ('k', 'v'):
                for layer in step[target + '_layers']:
                    stats[f'{target}_layer_{layer}'] = {'dense_bits': 100000, 'stored_bits': 100000 * (1 - achieved * (sampling['scoring_prefix'] / sampling['seq_len'] if step.get('prefill_only') else 1)),
                                                        'accounting': ACCOUNTING, 'updates': sampling['chunks'],
                                                        'controller_bound_hits': 0, 'max_final_target_deviation': 0.}
                    saving += achieved / (2 * run['model_shape']['layers'])
        nll = 1. + .01 * saving
        payload = {'simulation': identity(config, sampling, run),
                   'results': {'wikitext_chunks': {'token_perplexity,none': math.exp(nll)}},
                   'chunk_scores': [dict(chunk=i, nll=nll, tokens=sampling['seq_len']-sampling['scoring_prefix'])
                                    for i in range(sampling['offset'], sampling['offset'] + sampling['chunks'])],
                   'scoring': {'prefix_tokens': sampling['scoring_prefix']}, 'gate_stats': stats}
        if steps:
            payload['storage'] = {'scope':'simulated_prefill_cache_storage','targets': {'kv': {'compression': saving}}}
        write_json(config['output_path'], payload)


def fake_tasks(destination, run, *, summarizer):
    for config in run['configurations']:
        payload = {'simulation': simulation_identity(run, config, parse_kv_spec(config['kv'])),
                   'n-samples': {'ceval-valid': {'effective': 1346}},
                   'results': {'ceval-valid': {'acc,none': .5}}, 'gate_stats': {}}
        if config['kv']['pipeline']:
            payload['storage'] = {'targets': {'kv': {'compression': config['metadata']['global_budget']}}}
        write_json(config['output_path'], payload)
    summarizer(destination, run, require_complete=True)


class LadderTests(unittest.TestCase):
    def test_candidates_per_arm_and_algorithm(self):
        with tempfile.TemporaryDirectory() as folder:
            p = ladder.plan(Path(folder))
            torch.save({'weights': torch.ones(32, 8, 128)}, p['query_weights'])
            rows = ladder.candidates(p)
            self.assertEqual(len(rows), 64 * (6 + 9 + 9))
            for row in rows:
                step = row['kv']['pipeline'][0]
                self.assertEqual(step['method'], 'spatial_online')
                self.assertEqual(step['max_group'], 2 if row['arm'] == 'pairs' else 8)
                self.assertEqual(step['query_aware'], row['arm'] == 'query_hierarchy' and row['target'] == 'k')
                self.assertEqual(step['minimum_storage'], row['maximum_feasible'])
            offline = ladder.plan(Path(folder) / 'off', algorithm='offline', arms=('pairs', 'hierarchy'))
            rows = ladder.candidates(offline)
            self.assertEqual(len(rows), 64 * (5 + 9))
            self.assertTrue(all(r['kv']['pipeline'][0]['method'] == 'spatial_offline' for r in rows))
            self.assertNotIn('control', offline)

    def test_validation_gate_and_endpoint(self):
        rows = [dict(arm=a, budget=b, actual_saving=.491 if a == 'pairs' and b == .5 else b)
                for a in ladder.ARMS for b in ladder.BUDGETS]
        ladder.check_validation(rows, ladder.BUDGETS, ladder.ARMS, tolerance=.01)
        rows[0]['actual_saving'] = .25
        with self.assertRaises(ValueError):
            ladder.check_validation(rows, ladder.BUDGETS, ladder.ARMS, tolerance=.01)

    def test_staged_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = ladder.plan(out, layers=1, screen=2, refine=2, validation=2)
            p['pair_local_budgets'] = [.3, .4, .49]
            p['local_budgets'] = [.3, .4, .5, .85]
            p['dense_ceval_source'] = str(out / 'missing.json')
            torch.save({'weights': torch.ones(1, 2, 128)}, p['query_weights'])
            with patch.object(study, 'execute_chunks', side_effect=lambda path: fake_chunks(path, endpoint_saving=.491)):
                for stage in ('screen', 'refine', 'allocate', 'validate'):
                    ladder.run_stage(p, stage)
            selections = json.loads((out / 'selections.json').read_text())
            self.assertEqual(len(selections), 9)
            endpoint = next(s for s in selections if s['maximum_feasible'])
            self.assertAlmostEqual(endpoint['compression'], .491)
            for selection in selections:
                self.assertTrue(all(row['arm'] == selection['arm'] for row in selection['assignment']))
            with patch.object(study, 'execute_tasks', side_effect=fake_tasks):
                ladder.run_stage(p, 'tasks')
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['completed'], 10)
            self.assertEqual(sum(r.get('maximum_feasible', False) for r in report['rows']), 1)
            validation = json.loads((out / 'validation.json').read_text())
            validation[1]['actual_saving'] = .2
            write_json(out / 'validation.json', validation)
            with patch.object(study, 'execute_tasks') as execute:
                with self.assertRaisesRegex(RuntimeError, 'byte gate failed'):
                    ladder.run_stage(p, 'tasks')
                execute.assert_not_called()


class FixedRungTests(unittest.TestCase):
    def test_candidates_and_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = fixed_rungs.plan(Path(folder), layers=1)
            p['dense_ceval_source'] = str(Path(folder) / 'missing.json')
            rows = fixed_rungs.candidates(p)
            self.assertTrue(all(r['residual_entries'] == 0 for r in rows if r['arm'] == 'no_residual'))
            self.assertEqual({r['residual_entries'] for r in rows if r['arm'] == 'fixed_residual'}, {0, 16, 32})
            self.assertTrue(all(r['kv']['pipeline'][0]['fixed_residual'] == r['residual_entries'] for r in rows))
            with patch.object(study, 'execute_chunks', side_effect=fake_chunks):
                for stage in ('screen', 'refine', 'allocate', 'validate'):
                    fixed_rungs.run_stage(p, stage)
            selections = json.loads((Path(folder) / 'selections.json').read_text())
            self.assertEqual(len(selections), 8)
            for selection in selections:
                self.assertEqual(len(selection['kv']['pipeline']), 2)


class StudyTests(unittest.TestCase):
    def test_chunk_splits(self):
        p = ladder.plan(Path('/tmp/plan'))
        screen = study.chunk_run(p, [], 'screen')['sampling']
        refine = study.chunk_run(p, [], 'refine')['sampling']
        self.assertEqual((screen['split'], refine['split']), ('train', 'train'))
        self.assertEqual(refine['offset'], screen['chunks'])
        self.assertEqual(study.chunk_run(p, [], 'validation')['sampling']['split'], 'validation')

    def test_dense_reuse_requires_identity_and_full_question_count(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = ladder.plan(out, layers=1)
            source = out / 'source.json'
            p['dense_ceval_source'] = str(source)
            config = study.configuration('dense_ceval', {'pipeline': []}, out / 'tasks', study.CEVAL_VALID_5SHOT)
            base = {'model': 'hf', 'model_args': p['model_args'], 'batch_size': 1}
            payload = {'simulation': simulation_identity(base, config, parse_kv_spec(config['kv'])),
                       'n-samples': {'ceval-valid': {'effective': 1346}}, 'results': {'ceval-valid': {'acc,none': .5}}}
            stale = deepcopy(payload)
            stale['simulation']['num_fewshot'] = 0
            write_json(source, stale)
            study.task_run(p, [])
            self.assertFalse((out / 'tasks/dense_ceval.json').exists())
            write_json(source, payload)
            study.task_run(p, [])
            self.assertEqual(json.loads((out / 'tasks/dense_ceval.json').read_text()), payload)

    def test_live_single_slot_measurement(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        from compression_topics.spatial.algorithms.methods import SpatialConfig
        torch.manual_seed(13)
        torch.set_num_threads(2)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=160,
                                             num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                             head_dim=128)).eval()
        for algorithm in ('online', 'offline'):
            policy = SpatialConfig(algorithm=algorithm).pipeline(.4, k_layers=[0], v_layers=[])
            payload = evaluate_chunks(model, [torch.randint(0, 32, (72,))], policy, [1], scoring_prefix=64)
            measured = study.slot_measurement(payload)
            self.assertGreater(measured['compression'], .2)


if __name__ == '__main__':
    unittest.main()
