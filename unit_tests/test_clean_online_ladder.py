import tempfile
from pathlib import Path
import unittest
import torch
from compression_topics.spatial.scripts import clean_online_ladder as study
from compression_topics.spatial.algorithms import hierarchy

class CleanLadderTests(unittest.TestCase):
    def test_three_policies_and_common_pair_menu(self):
        with tempfile.TemporaryDirectory() as folder:
            p=study.plan(Path(folder))
            torch.save({'weights':torch.ones(32,8,128)},p['query_weights'])
            rows=study.candidates(p)
            self.assertEqual(len(rows),1536)
            for row in rows:
                step=row['kv']['pipeline'][0]
                self.assertEqual(step['head_decisions'],'per_head')
                self.assertEqual(step['mask_layout'],'separate')
                self.assertEqual(step['importance'],'none')
                self.assertEqual(step['max_group'],2 if row['arm']=='pairs' else 8)
                aware=row['arm']=='query_hierarchy' and row['target']=='k'
                self.assertEqual(step['distortion'],'query' if aware else 'relative')
                self.assertEqual(step['select_by'],'query' if aware else 'deviation')
    def test_validation_endpoint_and_matched_gates(self):
        rows=[dict(arm=a,budget=b,actual_saving=.491 if a=='pairs' and b==.5 else b) for a in study.ARMS for b in study.BUDGETS]
        study.check_validation(rows,study.BUDGETS,study.ARMS,tolerance=.01)
        rows[0]['actual_saving']=.25
        with self.assertRaises(ValueError):study.check_validation(rows,study.BUDGETS,study.ARMS,tolerance=.01)
    def test_independent_calibration_allocation_and_endpoint(self):
        import json
        from unittest.mock import patch
        from unit_tests.test_hierarchy_accuracy import HierarchyAccuracyTests
        fixture=HierarchyAccuracyTests()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);p=study.plan(out,layers=1)
            p['pair_local_budgets']=[.3,.4,.49];p['local_budgets']=[.3,.4,.5,.85]
            torch.save({'weights':torch.ones(1,2,128)},p['query_weights'])
            def execute(path):
                fixture.fake_chunks(path)
                run=json.loads(Path(path).read_text())
                for config in run['configurations']:
                    steps=config['kv']['pipeline']
                    if not any(step.get('saturate') for step in steps):continue
                    dest=Path(config['output_path']);payload=json.loads(dest.read_text())
                    for stat in payload['gate_stats'].values():stat['stored_bits']=stat['dense_bits']*.509
                    payload['storage']['targets']['kv']['compression']=.491
                    dest.write_text(json.dumps(payload))
            with patch.object(study.base.workflow,'execute_chunks',side_effect=execute):
                for stage in ('screen','refine','allocate','validate'):study.run_stage(p,stage)
            selections=json.loads((out/'selections.json').read_text())
            self.assertEqual(len(selections),9)
            endpoint=next(s for s in selections if s['maximum_feasible'])
            self.assertAlmostEqual(endpoint['compression'],.491)
            for selection in selections:
                self.assertTrue(all(row['arm']==selection['arm'] or row['arm']==study.base.ARM for row in selection['assignment']))
            run=study.tasks(p,selections)
            self.assertEqual(len(run['configurations']),10)
            self.assertEqual(sum(c.get('metadata',{}).get('maximum_feasible',False) for c in run['configurations']),1)

    def test_saturation_uses_minimum_pair_storage(self):
        torch.set_num_threads(2)
        x=torch.randn(1,2,128,128)
        table=hierarchy.build_table(x,target='v',max_group=2)
        result=hierarchy.run(table,.5,saturate=True)
        count=2*64
        expected=count*(16*(128+2))+16+8*((count*3+7)//8)
        self.assertEqual(result['stored_bits'],expected)
        self.assertTrue(all(size==2 and residual==0 for r in result['history'] for head in r['head_segments'] for _,size,residual in head))

if __name__=='__main__':unittest.main()
