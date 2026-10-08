"""Synthetic checks for independent-residual group compression."""
import unittest
import json
import math
import tempfile
from pathlib import Path
import torch
from compression_topics.spatial.algorithms import pair_gate
from compression_topics.spatial.algorithms.group_calibrated import apply, group_saving, merged_bits, metadata_bits
from compression_topics.spatial.scripts import group_calibrated_study as study
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec


class GroupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        pair_gate.reset_stats()

    def test_budget_norms_tail_and_decode(self):
        x = torch.randn(1, 2, 103, 128)
        for size in (2, 3, 4):
            for target in ('k', 'v'):
                pair_gate.reset_stats()
                y = apply(x, layer_idx=0, target=target, saving=.2,
                          group_size=size, residual_entries=16,
                          rope_tables=RopeTables(10000., 128))
                torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
                tail = x.shape[-2] // size * size
                self.assertTrue(torch.equal(y[..., tail:, :], x[..., tail:, :]))
                stats = study.measurement({'gate_stats': pair_gate.pop_stats()})
                self.assertGreaterEqual(stats['compression'], .2)
                self.assertLess(stats['compression'], .2 + .02)
                self.assertTrue(torch.equal(apply(x, layer_idx=0, target=target,
                    saving=.2, group_size=size, seq_start=1), x))

    def test_key_alignment_of_equal_unrotated_directions(self):
        rope = RopeTables(10000., 128)
        raw = torch.randn(1, 2, 1, 128).expand(1, 2, 12, 128).clone()
        raw *= torch.arange(1, 13).reshape(1, 1, 12, 1)
        cos, sin = rope.cos_sin(torch.arange(12), torch.float32)
        keys = apply_rope(raw, cos, sin, inverse=False)
        for size in (2, 3, 4):
            actual = apply(keys, layer_idx=0, target='k', saving=group_saving(0, group_size=size),
                           group_size=size, rope_tables=rope)
            torch.testing.assert_close(actual, keys, atol=2e-5, rtol=2e-5)

    def test_four_separate_residuals_match_manual_reconstruction(self):
        x = torch.randn(1, 1, 4, 128)
        unit = torch.nn.functional.normalize(x, dim=-1)
        mean = unit.mean(-2, keepdim=True)
        error = unit - mean
        ids = error.abs().topk(32, dim=-1).indices
        sparse = torch.zeros_like(error).scatter(-1, ids, error.gather(-1, ids))
        expected = torch.nn.functional.normalize(mean + sparse, dim=-1) * x.norm(dim=-1, keepdim=True)
        actual = apply(x, layer_idx=0, target='v', saving=.4, residual_entries=32)
        torch.testing.assert_close(actual, expected)

    def test_invalid_budget_and_zero_groups(self):
        x = torch.zeros(1, 1, 8, 128)
        with self.assertRaises(ValueError):
            apply(x, layer_idx=0, target='v', saving=.6, residual_entries=32)
        actual = apply(x, layer_idx=0, target='v', saving=.5)
        self.assertTrue(torch.equal(actual, x))
        self.assertEqual(pair_gate.pop_stats()['v_layer_0']['shortfall_updates'], 1)

    def test_exact_metadata_storage_and_budget_rounding(self):
        # 16 groups, one mean + four residuals + norms per merged group.
        x = torch.randn(1, 1, 64, 128)
        apply(x, layer_idx=0, target='v', saving=.3, residual_entries=16)
        stats = pair_gate.pop_stats()['v_layer_0']
        expected = (16 - stats['merged']) * 4 * 128 * 16
        expected += stats['merged'] * merged_bits(16) + metadata_bits(16)
        self.assertEqual(stats['stored_bits'], expected)
        self.assertEqual(stats['metadata_bits'], 32)
        self.assertEqual(stats['norm_bits'], stats['merged'] * 4 * 16)
        self.assertEqual(stats['residual_mask_bits'], stats['merged'] * 4 * 128)
        self.assertGreaterEqual(1 - expected / stats['dense_bits'], .3)
        # Bitmap/header remain present even when all groups stay dense.
        pair_gate.reset_stats()
        apply(torch.zeros_like(x), layer_idx=0, target='v', saving=.3)
        stats = pair_gate.pop_stats()['v_layer_0']
        self.assertEqual(stats['stored_bits'], stats['dense_bits'] + metadata_bits(16))

    def test_manifest_accounting_and_registration(self):
        manifest = study.plan(Path('/tmp/group-test'), layers=1)
        self.assertTrue(manifest['residual_mask_counted'])
        self.assertTrue(manifest['norm_overhead_counted'])
        self.assertTrue(manifest['flags_counted'])
        self.assertEqual(manifest['group_sizes'], [2, 4])
        for row in manifest['candidates']:
            self.assertLessEqual(row['budget'], group_saving(row['residual_entries'], group_size=row['group_size']))
            parse_kv_spec(row['kv'])

    def test_screen_refines_each_size_and_policy_chooses_per_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = study.plan(out, layers=2, residuals=(0, 4), local_budgets=(.1, .2, .3, .4, .5, .6))
            dense = Path(manifest['run']['configurations'][0]['output_path'])
            dense.parent.mkdir(parents=True)
            dense.write_text(json.dumps({'results': {'wikitext_chunks': {'token_perplexity,none': 10.}}}))
            for row in manifest['candidates']:
                preferred = 2 if row['layer'] == 0 else 4
                penalty = .01 if row['group_size'] == preferred else .3
                ppl = 10 + row['budget'] + penalty + row['residual_entries'] * .001
                stats = {'dense_bits': 100000, 'stored_bits': 100000 * (1-row['budget']),
                         'pairs': 100, 'merged': 60, 'cosine_sum': 59., 'cosine_min': .95,
                         'cutoff_sum': .95, 'updates': 1, 'shortfall_updates': 0}
                Path(row['output_path']).write_text(json.dumps({'simulation': {'kv': row['kv']},
                    'results': {'wikitext_chunks': {'token_perplexity,none': ppl}},
                    'chunk_scores': [{'chunk': i, 'nll': math.log(ppl)} for i in range(4)],
                    'gate_stats': {f"{row['target']}_layer_{row['layer']}": stats}}))
            calibrated = study.select(manifest)
            finalists = [r for r in calibrated['shortlisted'] if r['layer'] == 0 and r['target'] == 'k' and r['budget'] == .2]
            self.assertEqual({r['group_size'] for r in finalists}, {2, 4})
            study.comparisons(calibrated, calibrated)
            mixed = study.policy_calibration(calibrated, 'mixed')
            self.assertEqual(next(r for r in mixed['selected'] if r['layer'] == 0 and r['target'] == 'k' and r['budget'] == .2)['group_size'], 2)
            self.assertEqual(next(r for r in mixed['selected'] if r['layer'] == 1 and r['target'] == 'k' and r['budget'] == .2)['group_size'], 4)
            selections, run = study.task_plan(calibrated, 2, out)
            self.assertEqual(len(selections), 20)
            for selection in selections:
                slots = [(r['layer'], r['target']) for r in selection['assignment']]
                self.assertEqual(len(slots), len(set(slots)))
                self.assertGreaterEqual(selection['compression'] + 1e-12, selection['budget'])
                if selection['policy'] == 'pairs':
                    self.assertTrue(all(r['group_size'] == 2 for r in selection['assignment']))
                parse_kv_spec(selection['kv'])


if __name__ == '__main__':
    unittest.main()
