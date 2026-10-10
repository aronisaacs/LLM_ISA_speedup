import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from compression_topics.spatial.scripts import hierarchy_pilot as pilot
from engine.eval_runner.files import write_json


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);torch.set_num_threads(2)

    def test_complete_pilot_writes_preliminary_timing_and_matched_comparisons(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);captures=root/'captures';out=root/'out'
            write_json(captures/'plan.json', {'model_args':'pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16',
                'sampling':{'pool_chunks':80,'seed':0,'chunks':16,'seq_len':2048}})
            write_json(captures/'rope.json', {'rope_theta':500000.,'head_dim':128})
            (captures/'keys').mkdir()
            x=torch.randn(32,8,64,128).to(torch.bfloat16)
            torch.save({'keys':x,'values':x,'chunk':7},captures/'keys/chunk_00007.pt')
            argv=['pilot','--captures',str(captures),'--out',str(out),'--chunks','1',
                  '--layers','0','--prefixes','32','64','--budgets','.3','.6','--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):pilot.main()
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['rows']),40)
            failed=[r for r in result['rows'] if r['status']=='unreachable']
            self.assertTrue(failed);self.assertTrue(all(r['variant']=='pairs' and r['target_saving']==.6 for r in failed))
            self.assertEqual(len(list((out/'timings').glob('*.json'))),4)
            self.assertTrue((out/'preliminary.json').exists())
            self.assertTrue(result['comparisons'])
            self.assertEqual(len(result['format_comparisons']),8)
            for row in result['format_comparisons']:
                self.assertEqual(row['saved_mask_bytes'],8*(row['group_size']-1)*128/8)
            for row in result['rows']:
                if row['status']=='complete':
                    self.assertAlmostEqual(row['measured_saving'],1-row['stored_bits']/row['dense_bits'])

    def test_unmatched_bytes_are_flagged_instead_of_claimed_as_quality_gain(self):
        metadata=dict(chunk=1,layer=0,target='v',tokens=128,target_saving=.4,status='complete')
        rows=[dict(metadata,variant='pairs',measured_saving=.3,mean_relative_squared_error=.1),
              dict(metadata,variant='pairs_quads_separate',measured_saving=.4,mean_relative_squared_error=.05)]
        comparison=pilot.comparisons(rows)[0]
        self.assertFalse(comparison['matched_within_one_point'])
        self.assertAlmostEqual(comparison['saving_gap_points'],10.)

if __name__=='__main__':unittest.main()
