"""Budget calibration checks; synthetic data only, no model or network."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from compression_topics.spatial.algorithms import pair_gate
from compression_topics.spatial.algorithms.pair_calibrated import apply, pair_saving
from compression_topics.spatial.scripts.pair_calibrated_study import plan, select, allocate, task_plan, measurement
from engine.kv_compress.rope import RopeTables
from engine.kv_compress.spec import parse_kv_spec


class CompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.randn(1, 2, 101, 128)
        self.rope = RopeTables(10000., 128)
        pair_gate.reset_stats()

    def test_budget_norms_and_cosine_for_both_targets(self):
        for target in ('k', 'v'):
            for kept in (0, 4, 8, 16, 32):
                pair_gate.reset_stats()
                out = apply(self.x, layer_idx=3, target=target, saving=.25,
                            residual_entries=kept, rope_tables=self.rope)
                torch.testing.assert_close(out.norm(dim=-1), self.x.norm(dim=-1), atol=1e-5, rtol=1e-5)
                self.assertTrue(torch.equal(out[..., -1, :], self.x[..., -1, :]))
                measured = measurement({'gate_stats': pair_gate.pop_stats()})
                self.assertGreaterEqual(measured['compression'], .25)
                self.assertLess(measured['compression'], .25 + .005)
                self.assertTrue(-1 <= measured['min_reconstruction_cosine'] <= 1)

    def test_values_need_no_rope_and_decode_is_unchanged(self):
        apply(self.x, layer_idx=0, target='v', saving=.1)
        self.assertTrue(torch.equal(apply(self.x, layer_idx=0, target='k', saving=.1, seq_start=1), self.x))
        with self.assertRaises(ValueError):
            apply(self.x, layer_idx=0, target='k', saving=.1)

    def test_unreachable_budget_is_rejected(self):
        self.assertAlmostEqual(pair_saving(32), .34375)
        with self.assertRaises(ValueError):
            apply(self.x, layer_idx=0, target='v', saving=.4, residual_entries=32)

    def test_different_layer_residual_stats_do_not_mix(self):
        for layer, kept in ((0, 0), (1, 32)):
            apply(self.x, layer_idx=layer, target='v', saving=.25, residual_entries=kept)
        stats = pair_gate.pop_stats()
        self.assertEqual(set(stats), {'v_layer_0', 'v_layer_1'})
        self.assertEqual(stats['v_layer_0']['kept'], 0)
        self.assertEqual(stats['v_layer_1']['kept'], 32)


class CalibrationTests(unittest.TestCase):
    def test_plan_is_independent_and_excludes_infeasible_sizes(self):
        manifest = plan(Path('/tmp/calibration'), layers=2)
        for row in manifest['candidates']:
            step = row['kv']['pipeline'][0]
            self.assertEqual(len(step['k_layers']) + len(step['v_layers']), 1)
            self.assertLessEqual(row['budget'], pair_saving(row['residual_entries']))
            parse_kv_spec(row['kv'])
        self.assertEqual(set(row['target'] for row in manifest['candidates']), {'k', 'v'})

    def test_best_residual_is_chosen_per_slot_and_global_budgets_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = plan(out, layers=2, local_budgets=(.1, .2, .3, .4, .5), residuals=(0, 16))
            dense_path = Path(manifest['run']['configurations'][0]['output_path'])
            dense_path.parent.mkdir(parents=True)
            dense_path.write_text(json.dumps({'results': {'wikitext': {'word_perplexity,none': 10.}}}))
            for row in manifest['candidates']:
                preferred = (row['target'] == 'k' and row['residual_entries'] == 16)
                ppl = 10 + row['budget'] * (1 if preferred else 2) + row['layer'] * .01
                key = f"{row['target']}_layer_{row['layer']}"
                stats = {'dense_bits': 100000, 'stored_bits': 100000 * (1-row['budget']),
                         'pairs': 100, 'merged': 60, 'cosine_sum': 59., 'cosine_min': .95,
                         'cutoff_sum': .95, 'updates': 1, 'shortfall_updates': 0}
                Path(row['output_path']).write_text(json.dumps({'simulation': {'kv': row['kv']},
                    'results': {'wikitext': {'word_perplexity,none': ppl}}, 'gate_stats': {key: stats}}))
            calibrated = select(manifest)
            winner = next(r for r in calibrated['selected'] if r['layer'] == 0 and r['target'] == 'k' and r['budget'] == .2)
            self.assertEqual(winner['residual_entries'], 16)
            selections, run = task_plan(calibrated, 2, out)
            self.assertEqual(len(selections), 8)
            self.assertEqual(len(run['configurations']), 9)
            for selection in selections:
                self.assertGreaterEqual(selection['compression'] + 1e-12, selection['budget'])
                parse_kv_spec(selection['kv'])
                if selection['mode'] == 'keys':
                    self.assertTrue(all(not step['v_layers'] for step in selection['kv']['pipeline']))
                    self.assertAlmostEqual(selection['whole_kv_compression'], selection['compression']/2)

    def test_greedy_uses_cost_per_byte_and_reports_unreachable(self):
        def row(layer, target, compression, ppl):
            return {'layer': layer, 'target': target, 'budget': compression,
                    'compression': compression, 'ppl': ppl, 'kv': {'pipeline': [{'layer': layer}]}}
        calibrated = {'dense_ppl': 10., 'selected': [row(0,'k',.4,10.1), row(1,'k',.4,12.)]}
        chosen = allocate(calibrated, 2, ('k',), .1)
        self.assertEqual(chosen['assignment'][0]['layer'], 0)
        with self.assertRaises(ValueError):
            allocate(calibrated, 2, ('k',), .5)


if __name__ == '__main__':
    unittest.main()
