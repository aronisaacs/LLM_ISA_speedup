"""Clean ladder: alignment, residuals/norms, weighting, bytes and fresh protocol."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from compression_topics.spatial.algorithms import presentation_spatial as kernel
from compression_topics.spatial.scripts import presentation_ladder_study as study
from engine.eval_runner.chunks import evaluate_chunks
from engine.eval_runner.files import write_json
from engine.kv_compress import metrics
from engine.kv_compress.rope import RopeTables, apply_rope


class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        metrics.reset_stats()
        self.rope = RopeTables(10000., 128)

    def test_aligned_raw_pairs_do_not_merge_across_heads(self):
        raw = torch.randn(1, 2, 1, 128).expand(1, 2, 64, 128).clone()
        cos, sin = self.rope.cos_sin(torch.arange(64), torch.float32)
        keys = apply_rope(raw, cos, sin, inverse=False)
        out = kernel.apply(keys, layer_idx=0, target='k', saving=.3, directional=False, rope_tables=self.rope)
        torch.testing.assert_close(out, keys, atol=2e-5, rtol=2e-5)
        self.assertEqual(metrics.STATS['k_layer_0']['norm_bits'], 0)
        with self.assertRaisesRegex(ValueError, 'RoPE'):
            kernel.apply(keys, layer_idx=0, target='k', saving=.3, directional=False)

    def test_norms_residual_masks_tails_and_dense_decode(self):
        x = torch.randn(1, 2, 67, 128)
        for size in (2, 4):
            metrics.reset_stats()
            y = kernel.apply(x, target='v', layer_idx=0, saving=.3, residual_entries=16, group_size=size)
            torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=2e-5, rtol=2e-5)
            full = x.shape[-2] // size * size
            self.assertTrue(torch.equal(y[..., full:, :], x[..., full:, :]))
            self.assertTrue(torch.equal(kernel.apply(x, target='v', layer_idx=0, saving=.3, seq_start=1), x))
            row = metrics.STATS['v_layer_0']
            expected = x.numel() * 16 - row['merged'] * (16 * size * 128 - kernel.merged_bits(16, group_size=size))
            expected += kernel.metadata_bits(row['pairs'])
            self.assertEqual(row['stored_bits'], expected)
            self.assertEqual(row['norm_bits'], 16 * size * row['merged'])
            self.assertEqual(row['residual_mask_bits'], 128 * row['merged'] * (1 if size == 2 else size))
            self.assertGreaterEqual(1 - row['stored_bits'] / row['dense_bits'], .3)

    def test_weighted_key_residuals_preserve_sensitive_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            weights = torch.ones(1, 1, 128)
            weights[..., 1] = weights[..., 65] = 1000.
            torch.save({'weights': weights}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            x = torch.zeros(1, 1, 64, 128)
            x[..., 2] = 4
            x[..., ::2, 0] = 1
            x[..., 1::2, 0] = -1
            x[..., ::2, 1] = .1
            x[..., 1::2, 1] = -.1
            common = dict(layer_idx=0, target='k', saving=.1, residual_entries=1, rope_tables=self.rope)
            plain = kernel.apply(x, **common)
            weighted = kernel.apply(x, select_by='query', query_weights=str(path), query_weights_sha256=digest, **common)
            # At least one merged pair keeps the small but important coordinate.
            self.assertLess(float((weighted[..., 1] - x[..., 1]).square().sum()),
                            float((plain[..., 1] - x[..., 1]).square().sum()))
            with self.assertRaisesRegex(ValueError, 'key weights'):
                kernel.apply(x, target='v', layer_idx=0, saving=.1, select_by='query')


class StudyTests(unittest.TestCase):
    def test_four_rungs_have_fresh_independent_allocation_and_consistent_rope(self):
        plan = study.study_plan(Path('/private/tmp/clean-ladder-plan'), layers=2)
        rows = study.candidates(plan, 'test-digest')
        self.assertEqual(plan['global_budgets'], [.1, .2, .3, .4, .5])
        for rung in range(1, 5):
            selected = [r for r in rows if rung in r['rungs']]
            self.assertTrue(selected)
            for row in selected:
                step = row['kv']['pipeline'][0]
                self.assertEqual(step['method'], 'presentation_spatial')
                self.assertNotIn('rope', step)  # Cannot disable mandatory key alignment.
                self.assertNotIn('importance', step)
                self.assertNotIn('menu', step)
                self.assertEqual(step['directional'], rung != 1)
                if rung == 1:
                    self.assertEqual(step['residual_entries'], 0)
                if rung < 4:
                    self.assertEqual(step['group_size'], 2)
                if rung >= 3 and row['target'] == 'k' and row['residual_entries']:
                    self.assertEqual(step['select_by'], 'query')
        cases = study.budget_cases(plan)
        self.assertEqual(len([c for c in cases if c['status'] == 'eligible']), 17)
        self.assertEqual([c['rung'] for c in cases if c['status'] == 'unattainable'], [1, 2, 3])
        self.assertTrue(all(c['budget'] == .5 for c in cases if c['status'] == 'unattainable'))
        screen, refine, val = (study.sampling(plan, stage) for stage in ('screen', 'refine', 'validate'))
        self.assertEqual(screen['split'], 'train')
        self.assertEqual(refine['split'], 'train')
        self.assertEqual(val['split'], 'validation')
        self.assertEqual(screen['offset'] + screen['chunks'], refine['offset'])

    def test_fifty_percent_is_allocated_only_for_quad_rung(self):
        plan = study.study_plan(Path('/private/tmp/clean-ladder-plan'), layers=2)
        calibration = {'candidates': [], 'dense_ppl': 10.}
        with patch.object(study, 'allocate', side_effect=lambda *args: {'budget': args[-1]}) as allocator:
            selections = study.select_allocations(plan, calibration)
        self.assertEqual(len(selections), 17)
        self.assertEqual([s['rung'] for s in selections if s['tag'].endswith('b50')], [4])
        self.assertEqual(sum(call.args[-1] == .5 for call in allocator.call_args_list), 1)

    def test_live_tiny_model_workflow_and_fresh_task_ledger(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(2)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=128)).eval()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            plan = study.study_plan(out, layers=2, local_budgets=(.1, .4, .49, .6), budgets=(.1,),
                                    residuals=(0, 8, 16), screen_chunks=2, refine_chunks=2, validation_chunks=2)
            plan['seq_len'] = 32
            torch.save({'weights': torch.ones(2, 1, 128)}, out / 'query_weights.pt')
            task_commands = []
            def execute_local(path):
                run = json.loads(path.read_text())
                sample = run['sampling']
                ids = list(range(sample['offset'], sample['offset'] + sample['chunks']))
                prompts = [(torch.arange(32) + i) % 64 for i in ids]
                for config in run['configurations']:
                    result = evaluate_chunks(model, prompts, config['kv'], ids)
                    result['simulation'] = {'kv': config['kv']}
                    write_json(config['output_path'], result)
            def fake_tasks(command, **kwargs):
                task_commands.append(command)
                run = json.loads((out / 'tasks_run.json').read_text())
                for config in run['configurations']:
                    task = 'ceval-valid' if 'ceval' in config['name'] else 'wikitext'
                    key = 'acc,none' if task == 'ceval-valid' else 'word_perplexity,none'
                    write_json(config['output_path'], {'results': {task: {key: .5 if task == 'ceval-valid' else 10.}},
                        'storage': {'targets': {'kv': {'compression': .1}}}})
            with patch.object(study, 'execute_chunks', side_effect=execute_local), \
                 patch.object(study.subprocess, 'run', side_effect=fake_tasks):
                for stage in study.STAGES[1:]:
                    study.run_stage(plan, stage)
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual({r['rung'] for r in report['rows']}, {1, 2, 3, 4})
            self.assertEqual(task_commands[0][task_commands[0].index('--results-root') + 1], str(out))
            selections = json.loads((out / 'selections.json').read_text())
            self.assertTrue(all(r['compression'] >= .1 for r in selections))

    def test_fresh_weight_capture_uses_training_data_and_restores_rotary_hook(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig, LlamaForCausalLM
        import transformers.models.llama.modeling_llama as llama
        torch.set_num_threads(2)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=128)).eval()
        original = llama.apply_rotary_pos_emb
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            plan = study.study_plan(out, layers=2, screen_chunks=2, refine_chunks=2)
            plan['seq_len'] = 32
            pool = study.sampling(plan, 'weights')['pool_chunks']
            prompts = [(torch.arange(32) + i) % 64 for i in range(pool)]
            with patch('engine.eval_runner.execute.load_model_if_needed',
                       return_value=(SimpleNamespace(model=model, tokenizer=object()), None, 'cpu', '')), \
                 patch.object(study, 'wikitext_chunks', return_value=(prompts, list(range(pool)))) as chunks:
                study.capture_weights(plan)
            self.assertIs(llama.apply_rotary_pos_emb, original)
            self.assertEqual(chunks.call_args.kwargs['split'], 'train')
            payload = torch.load(out / 'query_weights.pt', weights_only=True)
            self.assertEqual(tuple(payload['weights'].shape), (2, 1, 128))
            self.assertTrue(torch.isfinite(payload['weights']).all())
            self.assertEqual(payload['source']['sampling']['split'], 'train')
            self.assertEqual(len(payload['chunk_ids']), 4)


if __name__ == '__main__':
    unittest.main()
