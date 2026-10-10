"""Reusable local-calibration and whole-model allocation contracts."""
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from engine.eval_runner.files import write_json
from engine.layer_select.study import (measure_candidates, select_settings, allocate_settings,
                                       require_matched_savings)
from engine.layer_select.greedy.calibrated import choose_candidates

class CalibrationStudyTests(unittest.TestCase):
    def payload(self, kv, saving, nll=1.):
        return {'simulation': {'pretrained': 'test-model', 'kv': kv, 'sampling': {'split': 'train'}},
                'scoring': {'prefix': 32},
                'results': {'wikitext_chunks': {'token_perplexity,none': 3.}},
                'chunk_scores': [{'chunk': i, 'tokens': 16, 'nll': nll} for i in (1, 2)],
                'gate_stats': {'k_layer_0': {'dense_bits': 1000, 'stored_bits': 1000*(1-saving),
                    'pairs': 10, 'merged': 4, 'cosine_min': .9, 'cosine_sum': 3.6,
                    'cutoff_sum': .9, 'updates': 1, 'shortfall_updates': 0}}}

    def test_matched_calibration_keeps_target_measurement_and_source_separate(self):
        row = {'name': 'candidate', 'layer': 0, 'target': 'k', 'budget': .2,
               'kv': {'pipeline': [{'method': 'example', 'saving': .2}]}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dense = self.payload({'pipeline': []}, 0)
            payload = self.payload(row['kv'], .18, 1.1)
            write_json(folder/'dense.json', dense)
            write_json(folder/'candidate.json', payload)
            measured = measure_candidates([row], folder, tolerance=.03, budget_rule='absolute')
            result = measured['candidates'][0]
            self.assertEqual(result['target_saving'], .2)
            self.assertAlmostEqual(result['measured_saving'], .18)
            self.assertEqual(result['calibration_source']['chunks'], [1,2])
            self.assertEqual(result['calibration_source']['simulation']['pretrained'], 'test-model')
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                measure_candidates([row], folder, tolerance=.001, budget_rule='minimum')
            payload['chunk_scores'][0]['tokens'] = 15
            write_json(folder/'candidate.json', payload)
            with self.assertRaisesRegex(ValueError, 'samples differ'):
                measure_candidates([row], folder, tolerance=.03, budget_rule='absolute')
            payload['chunk_scores'][0]['tokens'] = 16
            payload['simulation']['pretrained'] = 'different-model'
            write_json(folder/'candidate.json', payload)
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                measure_candidates([row], folder, tolerance=.03, budget_rule='absolute')

    def test_method_selection_and_trim_do_not_mutate_local_measurements(self):
        rows = [{'name': f'{arm}_{target}', 'arm': arm, 'layer': 0, 'target': target,
                 'budget': .25, 'compression': .24, 'measured_saving': .24, 'ppl': ppl,
                 'kv': {'pipeline': [{'method': arm, 'saving': .25}]}}
                for arm, ppl in [('first', 11.), ('second', 10.5)] for target in ('k','v')]
        calibration = {'dense_ppl': 10., 'candidates': rows}
        original = deepcopy(calibration)
        settings = select_settings(calibration, include=lambda row: row['arm'] == 'first')
        assigned = allocate_settings(calibration, settings, 1, ('k','v'), .2, trim_overshoot=True)
        self.assertEqual(calibration, original)
        self.assertAlmostEqual(sum(r['allocation_saving'] for r in assigned['assignment'])/2, .2)
        self.assertTrue(all(r['measured_saving'] == .24 for r in assigned['assignment']))
        adjusted = next(r for r in assigned['assignment'] if 'allocation_adjustment' in r)
        self.assertFalse(adjusted['allocation_adjustment']['directly_calibrated'])
        self.assertEqual(adjusted['allocation_adjustment']['calibrated_target_saving'], .25)
        # Allocator must use its explicit assumption, not measured or stale aliases.
        selected,total = choose_candidates({(0,'k'): [{'compression': .9, 'allocation_saving': .2, 'ppl': 11}]}, 10, .2)
        self.assertEqual(total, .2)

    def test_validation_checks_all_methods_and_actual_savings(self):
        rows = [{'arm': arm, 'budget': .3, 'actual_saving': .3} for arm in ('fixed','online','offline')]
        require_matched_savings(rows, [.3], ('fixed','online','offline'), tolerance=.01)
        with self.assertRaisesRegex(ValueError, 'missing'):
            require_matched_savings(rows[:2], [.3], ('fixed','online','offline'), tolerance=.01)
        duplicate = [rows[0],rows[0],rows[2]]
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            require_matched_savings(duplicate, [.3], ('fixed','online','offline'), tolerance=.01)
        rows[2]['actual_saving'] = .32
        with self.assertRaisesRegex(ValueError, 'not matched'):
            require_matched_savings(rows, [.3], ('fixed','online','offline'), tolerance=.01)
        rows[2]['actual_saving'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'not matched'):
            require_matched_savings(rows, [.3], ('fixed','online','offline'), tolerance=.01)

if __name__ == '__main__':
    unittest.main()
