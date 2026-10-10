import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.eval_runner.timing import TimingRecord


class TimingTests(unittest.TestCase):
    def test_success_and_failure_are_distinct_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = dict(name='same_config', run='run.json', worker=2, identity={'kv': {}})
            with TimingRecord(root, **metadata) as timing:
                with timing.phase('evaluation'):
                    active = json.loads(timing.path.read_text())
                    self.assertEqual(active['active_phase'], 'evaluation')
                    self.assertEqual(active['status'], 'running')
            with self.assertRaisesRegex(ValueError, 'evaluation failed'):
                with TimingRecord(root, **metadata):
                    raise ValueError('evaluation failed')
            records = [json.loads(p.read_text()) for p in Path(root, 'timings').glob('*.json')]
            self.assertEqual(len(records), 2)
            self.assertEqual({r['status'] for r in records}, {'complete', 'failed'})
            success = next(r for r in records if r['status'] == 'complete')
            self.assertGreaterEqual(success['elapsed_seconds'], success['phases']['evaluation'])
            self.assertNotIn('active_phase', success)
            self.assertEqual(success['worker'], 2)

    def test_unwritable_telemetry_does_not_fail_evaluation(self):
        with patch('engine.eval_runner.timing.write_json', side_effect=OSError('disk full')):
            with TimingRecord('/unused', name='test', run='run', worker=0, identity={}) as timing:
                with timing.phase('evaluation'):
                    pass
            self.assertEqual(timing.data['status'], 'complete')

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            with TimingRecord(root, name='dry', run='run', worker=0, identity={}, enabled=False):
                pass
            self.assertFalse(Path(root, 'timings').exists())
