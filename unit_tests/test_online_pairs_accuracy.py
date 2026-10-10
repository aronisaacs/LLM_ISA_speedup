"""Independent calibration, allocation and full accuracy protocol guards."""
import tempfile
import unittest
from pathlib import Path
from compression_topics.spatial.scripts import online_pairs_accuracy as study
from engine.eval_runner.cache import simulation_identity
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec

class AccuracyTests(unittest.TestCase):
    def test_calibration_is_independent_and_allocation_matches_nominal_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=study.plan(Path(folder),layers=2,budgets=(.1,.2))
            rows=study.candidates(plan)
            measured=[]
            for row in rows:
                # Reverse layer preferences between methods to verify that
                # allocation actually uses each method's separate loss curves.
                preferred=0 if row['arm']=='fixed' else 1
                measured.append({**row,'compression':row['budget'],
                                 'ppl':10+row['budget']*(1 if row['layer']==preferred else 100)})
            calibration={'dense_ppl':10,'candidates':measured}
            selections=study.select_allocations(plan,calibration)
            for selection in selections:
                saving=sum(r['budget'] for r in selection['assignment'])/4
                self.assertAlmostEqual(saving,selection['budget'])
                self.assertTrue(all(r['arm']==selection['arm'] for r in selection['assignment']))
                self.assertEqual({r['layer'] for r in selection['assignment']},
                                 {0 if selection['arm']=='fixed' else 1})
            self.assertTrue(all(r['budget'] in plan['local_budgets'] for r in measured))
            for stage in ('screen','refine','validation'):
                run=study.chunk_run(plan,[],stage)
                self.assertEqual(run['sampling']['scoring_prefix'],1536)
                self.assertEqual(run['sampling']['split'],'validation' if stage=='validation' else 'train')
            self.assertEqual(study.chunk_run(plan,[],'refine')['sampling']['offset'],4)

    def test_full_protocol_summary_rejects_partial_samples_and_stale_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);plan=study.plan(out)
            selections=[{'tag':arm+'_b30','arm':arm,'budget':.3,'kv':{'pipeline':[study.candidates(plan)[0 if arm=='fixed' else -1]['kv']['pipeline'][0]]}} for arm in study.ARMS]
            run=study.task_run(plan,selections)
            for config in run['configurations']:
                self.assertEqual(config['tasks'],['ceval-valid'])
                self.assertEqual(config['num_fewshot'],5)
                self.assertTrue(config['apply_chat_template'])
                self.assertNotIn('limit',config)
            with self.assertRaisesRegex(ValueError,'incomplete'):study.summarize(out,run,require_complete=True)
            for config in run['configurations']:
                payload={'simulation':simulation_identity(run,config,parse_kv_spec(config['kv'])),
                         'results':{'ceval-valid':{'acc,none':.5}},
                         'n-samples':{'ceval-valid':{'effective':1346}},
                         'storage':{'targets':{'kv':{'compression':.3}}},'gate_stats':{}}
                write_json(config['output_path'],payload)
            report=study.summarize(out,run,require_complete=True)
            self.assertTrue(report['comparisons'][0]['matched_within_one_point'])
            payload['storage']['targets']['kv']['compression']=.33
            write_json(config['output_path'],payload)
            comparison=next(r for r in study.summarize(out,run)['comparisons'] if r['reference_arm']=='adaptive' and r['comparison_arm']=='offline')
            self.assertFalse(comparison['matched_within_one_point'])
            payload['n-samples']['ceval-valid']['effective']=16
            write_json(config['output_path'],payload)
            with self.assertRaisesRegex(ValueError,'1346'):study.summarize(out,run)
            payload['n-samples']['ceval-valid']['effective']=1346
            payload['simulation']['kv']={'pipeline':[]}
            write_json(config['output_path'],payload)
            with self.assertRaisesRegex(ValueError,'identity differs'):study.summarize(out,run)

if __name__=='__main__':unittest.main()
