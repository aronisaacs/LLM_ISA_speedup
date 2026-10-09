"""Live Q handoff, causal continuation scoring and shared-policy checks."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts import group_rd_integrated_study as study
from engine.eval_runner.chunks import evaluate_chunks
from engine.kv_compress.importance import attention_received
from engine.eval_runner.files import write_json


class ImportanceTests(unittest.TestCase):
    def test_tiled_attention_matches_explicit_causal_gqa(self):
        torch.manual_seed(7)
        q, k = torch.randn(1, 4, 7, 8), torch.randn(1, 2, 7, 8)
        actual = attention_received(q, k, block=2)
        scores = q[0] @ k[0].repeat_interleave(2, dim=0).transpose(-1, -2) / 8 ** .5
        mask = torch.arange(7)[None, :] > torch.arange(7)[:, None]
        probs = scores.masked_fill(mask, float('-inf')).softmax(-1)
        for part, rows in enumerate((list(range(7)), list(range(0, 7, 2)), list(range(1, 7, 2)))):
            expected = torch.zeros(4, 7)
            for token in range(7):
                visible = [r for r in rows if r >= token]
                if visible:
                    expected[:, token] = probs[:, visible, token].mean(-1)
            torch.testing.assert_close(actual[part], expected.reshape(2, 2, 7).sum(1))

    def test_importance_changes_memory_allocation_at_same_budget(self):
        torch.manual_seed(2)
        x = torch.randn(1, 1, 32, 128)
        weights = torch.cat((torch.full((1, 1, 16), 100.), torch.full((1, 1, 16), .01)), -1)
        opts = dict(rope_tables=None, menu=study.CANDIDATE_MENUS['flag_d'], distortion='squared')
        plain = group_rd.build_menu(x, **opts)
        weighted = group_rd.build_menu(x, token_weights=weights, **opts)
        budget = x.numel() * 16 * .6 - 40
        a = group_rd.select(plain, group_rd.solve_lambda(plain, budget))
        b = group_rd.select(weighted, group_rd.solve_lambda(weighted, budget))
        self.assertLessEqual(float(weighted.bits[b].sum()), budget)
        self.assertGreater(float(weighted.bits[b[..., :4]].sum()), float(plain.bits[a[..., :4]].sum()))

    def test_missing_importance_and_decode_rejected(self):
        x = torch.randn(1, 1, 16, 128)
        with self.assertRaisesRegex(ValueError, 'decode=False'):
            group_rd.apply(x, target='v', layer_idx=0, saving=.4, importance='prefill_attention')
        with self.assertRaisesRegex(ValueError, 'needs weights'):
            group_rd.apply(x, target='v', layer_idx=0, saving=.4, importance='prefill_attention', decode=False)


class LiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(11)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=128)).eval()
        cls.model.config._attn_implementation = 'eager'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / 'weights.pt'
        torch.save({'weights': torch.ones(2, 1, 128)}, self.weights)
        self.digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.spec = {'pipeline': [study.policy(layer, target, .4, self.weights, self.digest)
                                  for layer in range(2) for target in ('k', 'v')]}

    def test_hooks_restore_and_future_tokens_do_not_change_prefix_choices(self):
        import transformers.models.llama.modeling_llama as llama
        from transformers.cache_utils import Cache
        rotary, cache = llama.apply_rotary_pos_emb, Cache.update
        ids = torch.arange(24) % 64
        changed = ids.clone()
        changed[16:] = (changed[16:] + 20) % 64
        first = evaluate_chunks(self.model, [ids, ids], self.spec, [1, 2], scoring_prefix=16)
        second = evaluate_chunks(self.model, [changed], self.spec, [1], scoring_prefix=16)
        self.assertIs(llama.apply_rotary_pos_emb, rotary)
        self.assertIs(Cache.update, cache)
        self.assertEqual(first['storage']['targets']['kv']['compression'], second['storage']['targets']['kv']['compression'])
        for key in first['gate_stats']:
            self.assertEqual(first['gate_stats'][key]['stored_bits'], 2 * second['gate_stats'][key]['stored_bits'])
        self.assertEqual(first['chunk_scores'][0]['tokens'], 8)
        self.assertGreaterEqual(first['storage']['targets']['kv']['compression'], .399)

    def test_dense_continuation_matches_full_causal_forward(self):
        ids = torch.arange(24) % 64
        result = evaluate_chunks(self.model, [ids], {'pipeline': []}, [0], scoring_prefix=16)
        with torch.inference_mode():
            output = self.model(input_ids=ids[None], use_cache=False)
            expected = torch.nn.functional.cross_entropy(output.logits[:, 15:23].float().reshape(-1, 64), ids[16:])
        self.assertAlmostEqual(result['chunk_scores'][0]['nll'], float(expected), places=5)
        self.assertEqual(result['storage']['targets']['kv']['compression'], 0)

    def test_plan_uses_complete_policy_and_hash_guard(self):
        manifest = study.plan(Path(self.temp.name), layers=2, budgets=(.25, .4), weights_path=self.weights)
        self.assertEqual(len(manifest['candidates']), 8)
        for candidate in manifest['candidates']:
            step = candidate['kv']['pipeline'][0]
            self.assertEqual(step['importance'], 'prefill_attention')
            self.assertFalse(step['decode'])
            self.assertEqual(len(step['menu']), 8)
            if candidate['target'] == 'k':
                self.assertEqual(step['select_by'], 'query')
        with self.assertRaisesRegex(ValueError, 'changed'):
            group_rd._verified_weights(str(self.weights), 'wrong')

    def test_allocator_uses_new_policy_sensitivity_and_conservative_rates(self):
        rows = []
        for layer in range(2):
            for target in ('k', 'v'):
                for budget in (.25, .55):
                    rows.append({'layer': layer, 'target': target, 'budget': budget,
                                 'compression': budget + .02, 'ppl': 10 + budget ** 2 * (.01 if layer == 0 else 10),
                                 'kv': {'pipeline': [study.policy(layer, target, budget, self.weights, self.digest)]}})
        selection = study.allocate_policy({'dense_ppl': 10., 'selected': rows}, 2, .4)
        budgets = {(row['layer'], row['target']): row['budget'] for row in selection['assignment']}
        self.assertEqual(budgets, {(0, 'k'): .55, (0, 'v'): .55, (1, 'k'): .25, (1, 'v'): .25})
        self.assertAlmostEqual(selection['compression'], .4)
        self.assertAlmostEqual(selection['measured_calibration_compression'], .42)
        self.assertTrue(all(s['importance'] == 'prefill_attention' for s in selection['kv']['pipeline']))

    def test_fresh_curves_pool_screen_refine_and_reject_old_policy(self):
        out = Path(self.temp.name)
        manifest = study.plan(out, layers=2, budgets=(.4,), weights_path=self.weights)
        for stage, chunk_id in (('screen', 10), ('refine', 20)):
            write_json(out / stage / 'dense.json', {'simulation': {'kv': {'pipeline': []}},
                       'chunk_scores': [{'chunk': chunk_id, 'nll': 2., 'tokens': 8}]})
            for candidate in manifest['candidates']:
                write_json(out / stage / f"{candidate['name']}.json", {
                    'simulation': {'kv': candidate['kv']},
                    'chunk_scores': [{'chunk': chunk_id, 'nll': 2.01, 'tokens': 8}],
                    'gate_stats': {'slot_layer_0': {'dense_bits': 1000, 'stored_bits': 600,
                        'pairs': 4, 'merged': 2, 'cosine_min': .9, 'cosine_sum': 1.8,
                        'cutoff_sum': 0, 'updates': 1, 'shortfall_updates': 0}}})
        curves = study.measured_curves(manifest, 'refine')
        self.assertEqual([r['chunk'] for r in curves['selected'][0]['chunk_scores']], [10, 20])
        self.assertAlmostEqual(curves['selected'][0]['delta_nll'], .01)
        first = manifest['candidates'][0]
        path = out / 'refine' / f"{first['name']}.json"
        import json
        payload = json.loads(path.read_text())
        payload['simulation']['kv']['pipeline'][0]['importance'] = 'none'
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, 'stale policy'):
            study.measured_curves(manifest, 'refine')

    def test_staged_workflow_from_live_pilot_to_final_summary(self):
        from unittest.mock import patch
        import json
        out = Path(self.temp.name)
        manifest = study.plan(out, layers=2, budgets=(.25, .4), screen_chunks=2,
                              refine_chunks=2, validation_chunks=2, weights_path=self.weights)
        manifest.update(seq_len=32, scoring_prefix=24)

        def execute_local(path):
            run = json.loads(path.read_text())
            sampling = run['sampling']
            chosen = list(range(sampling['offset'], sampling['offset'] + sampling['chunks']))
            prompts = [(torch.arange(32) + i) % 64 for i in chosen]
            for config in run['configurations']:
                result = evaluate_chunks(self.model, prompts, config['kv'], chosen, scoring_prefix=24)
                result['simulation'] = {'kv': config['kv']}
                write_json(config['output_path'], result)

        def fake_ceval(*args, **kwargs):
            run = json.loads((out / 'tasks_run.json').read_text())
            for config in run['configurations']:
                write_json(config['output_path'], {'results': {'ceval-valid': {'acc,none': .5}},
                    'storage': {'targets': {'kv': {'compression': .4}}}})

        with patch.object(study, 'execute_chunks', side_effect=execute_local), \
             patch.object(study.subprocess, 'run', side_effect=fake_ceval):
            for stage in study.STAGES:
                study.run_stage(manifest, stage)
        report = json.loads((out / 'summary.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertGreaterEqual(report['rows'][0]['measured_prefill_kv_saving'], .399)
        self.assertTrue(json.loads((out / 'validation_gate.json').read_text())['rows'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
