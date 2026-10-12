import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from compression_topics.spatial.algorithms import windows,methods
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import fixed_hierarchy as study

class FixedHierarchyTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(4);torch.set_num_threads(2)
    def test_group_counts_restrict_choices(self):
        x=torch.randn(1,2,64,64)
        counts={'2':32,'4':16,'8':0}
        table,recon=windows.build(x,target='v',residual_counts=counts)
        for price in (.01,.1,1.,100.):
            choices=[table.choose(w,price) for w in range(len(table))]
            for choice in choices:
                for head in choice.heads:
                    self.assertTrue(all(r is None or r==counts[str(size)] for _,size,r in head.segments))
            self.assertTrue(torch.isfinite(windows.restore(x,choices,recon)).all())
    def test_prefill_only_keeps_continuation_dense(self):
        from unit_tests.test_spatial_algorithms import cache_run,correlated
        x=correlated(96)
        output,stats=cache_run(x,[64,32],dict(saving=.4,prefill_only=True,residual_counts={'2':16,'4':0,'8':0}))
        self.assertTrue(torch.equal(output[...,64:,:],x[...,64:,:]))
        self.assertFalse(torch.equal(output[...,:64,:],x[...,:64,:]))
        methods.reset()
    def test_configuration_rejects_invalid_combinations(self):
        for counts in ({'2':16},{'2':8,'4':0,'8':0},{'2':True,'4':0,'8':0}):
            with self.assertRaises(ValueError):SpatialConfig(residual_counts=counts)
    def test_staged_protocol_and_quality_gate_on_resume(self):
        from unit_tests.test_spatial_studies import fake_chunks,fake_tasks
        from engine.eval_runner.files import write_json
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);p=study.plan(out,layers=1);p['local_budgets']=[.1,.4,.55];p['dense_ceval_source']=str(out/'missing.json')
            ref=SpatialConfig(prefill_only=True).pipeline(.4,k_layers=[0],v_layers=[0])
            with patch.object(study,'reference_selection',return_value=ref),patch.object(study.base,'execute_chunks',side_effect=fake_chunks):
                for stage in ('screen','refine','allocate','validate'):study.run_stage(p,stage)
            with patch.object(study.base,'execute_tasks',side_effect=fake_tasks):study.run_stage(p,'tasks')
            report=study.base.load(out/'summary.json');self.assertEqual(report['completed'],2)
            self.assertIn('reference_comparison',report)
            bad=study.base.load(out/'validation/fixed_hierarchy_b40.json')
            bad['results']['wikitext_chunks']['token_perplexity,none']=100.
            write_json(out/'validation/fixed_hierarchy_b40.json',bad)
            with patch.object(study.base,'execute_tasks') as execute:
                with self.assertRaises(RuntimeError):study.run_stage(p,'tasks')
                execute.assert_not_called()
    def test_candidates_are_four_format_prefill_query_policies(self):
        p=study.plan(Path('/private/tmp/fixed-plan'));rows=study.candidates(p)
        self.assertEqual(len(p['residual_combinations']),12)
        for row in rows:
            step=row['kv']['pipeline'][0]
            self.assertTrue(step['prefill_only'])
            self.assertEqual(len(step['residual_counts']),3)
            self.assertEqual(step['query_aware'],row['target']=='k')
            self.assertEqual(step['max_group'],8)
if __name__=='__main__':unittest.main()
