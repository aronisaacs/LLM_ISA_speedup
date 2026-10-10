"""New accuracy study keeps calibration, allocation, validation and tasks aligned."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compression_topics.spatial.scripts import hierarchy_accuracy as study
from engine.eval_runner.cache import simulation_identity
from engine.eval_runner.chunks import identity
from engine.eval_runner.files import write_json
from engine.kv_compress import parse_kv_spec


class HierarchyAccuracyTests(unittest.TestCase):
    def test_default_plan_has_only_online_separate_mask_eight_capable_policies(self):
        plan=study.plan(Path('/tmp/plan'))
        rows=study.candidates(plan)
        self.assertEqual(len(rows),448)
        self.assertEqual(plan['budgets'],[n/10 for n in range(1,9)])
        for row in rows:
            spec=parse_kv_spec(row['kv'])
            step=spec.pipeline[0]
            self.assertEqual(step.kwargs['decision'],'online')
            self.assertEqual(step.kwargs['max_group'],8)
            self.assertEqual(step.kwargs['mask_layout'],'separate')
        screen=study.workflow.chunk_run(plan,[],'screen')['sampling']
        refine=study.workflow.chunk_run(plan,[],'refine')['sampling']
        self.assertEqual(screen['split'],'train');self.assertEqual(refine['split'],'train')
        self.assertEqual(refine['offset'],screen['chunks'])
        self.assertEqual(study.workflow.chunk_run(plan,[],'validation')['sampling']['split'],'validation')

    def fake_chunks(self,path):
        run=json.loads(Path(path).read_text());sampling=run['sampling']
        for config in run['configurations']:
            steps=config['kv']['pipeline'];stats={};saving=0.
            for step in steps:
                for target in ('k','v'):
                    for layer in step[target+'_layers']:
                        stats[f'{target}_layer_{layer}']={'dense_bits':100000,'stored_bits':100000*(1-step['saving']),
                            'accounting':study.accounting('separate'),'updates':sampling['chunks'],
                            'controller_bound_hits':0,'max_final_target_deviation':0.}
                        saving+=step['saving']/2  # fixture has one layer and two slots
            nll=1.+.01*saving
            payload={'simulation':identity(config,sampling,run),'results':{'wikitext_chunks':{'token_perplexity,none':math.exp(nll)}},
                'chunk_scores':[dict(chunk=i,nll=nll,tokens=16) for i in range(sampling['offset'],sampling['offset']+sampling['chunks'])],
                'scoring':{'prefix_tokens':sampling['scoring_prefix']},'gate_stats':stats}
            if steps:payload['storage']={'targets':{'kv':{'compression':saving}}}
            write_json(config['output_path'],payload)

    def test_complete_protocol_reuses_runner_and_gates_resumed_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);plan=study.plan(out,layers=1,budgets=(.1,.4),screen=2,refine=2,validation=2)
            plan['local_budgets']=[.1,.4,.85];plan['dense_ceval_source']=str(out/'missing.json')
            with patch.object(study.workflow,'execute_chunks',side_effect=self.fake_chunks):
                for stage in ('screen','refine','allocate','validate'):study.run_stage(plan,stage)
            cal=json.loads((out/'calibration.json').read_text())
            self.assertEqual(len(cal['candidates']),6)
            self.assertNotIn('merged_fraction',cal['candidates'][0])
            selections=json.loads((out/'selections.json').read_text())
            self.assertTrue(all(s['allocation_basis']=='measured_calibration_bytes' for s in selections))
            for selection in selections:
                for row in selection['assignment']:
                    self.assertEqual(row['kv']['pipeline'][0]['saving'],row['target_saving'])
                    self.assertNotIn('allocation_adjustment',row)
            def fake_tasks(destination,run,*,summarizer):
                for config in run['configurations']:
                    payload={'simulation':simulation_identity(run,config,parse_kv_spec(config['kv'])),
                             'n-samples':{'ceval-valid':{'effective':1346}},
                             'results':{'ceval-valid':{'acc,none':.5}},'gate_stats':{}}
                    if config['kv']['pipeline']:
                        payload['storage']={'targets':{'kv':{'compression':config['metadata']['global_budget']}}}
                    write_json(config['output_path'],payload)
                summarizer(destination,run,require_complete=True)
            with patch.object(study.workflow,'execute_tasks',side_effect=fake_tasks):study.run_stage(plan,'tasks')
            report=json.loads((out/'summary.json').read_text())
            self.assertEqual(report['completed'],3)
            self.assertEqual(report['comparisons'],[])
            self.assertEqual(report['rows'][0]['actual_prefill_kv_saving'],0.)
            self.assertTrue(all(r['within_one_point'] for r in report['rows'][1:]))
            validation=json.loads((out/'validation.json').read_text());validation[0]['actual_saving']=.2
            write_json(out/'validation.json',validation)
            with patch.object(study.workflow,'execute_tasks') as execute:
                with self.assertRaisesRegex(RuntimeError,'byte gate failed'):study.run_stage(plan,'tasks')
                execute.assert_not_called()

    def test_dense_reuse_requires_exact_identity_and_full_question_count(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);plan=study.plan(out,layers=1,budgets=(.1,))
            source=out/'source.json';plan['dense_ceval_source']=str(source)
            config=study.configuration('dense_ceval',{'pipeline':[]},out/'tasks',study.CEVAL_VALID_5SHOT)
            base={'model':'hf','model_args':plan['model_args'],'batch_size':1}
            payload={'simulation':simulation_identity(base,config,parse_kv_spec(config['kv'])),
                     'n-samples':{'ceval-valid':{'effective':1346}},'results':{'ceval-valid':{'acc,none':.5}}}
            stale=deepcopy(payload);stale['simulation']['num_fewshot']=0
            write_json(source,stale);study.task_run(plan,[])
            self.assertFalse((out/'tasks/dense_ceval.json').exists())
            partial=deepcopy(payload);partial['n-samples']['ceval-valid']['effective']=16
            write_json(source,partial);study.task_run(plan,[])
            self.assertFalse((out/'tasks/dense_ceval.json').exists())
            write_json(source,payload);study.task_run(plan,[])
            self.assertEqual(json.loads((out/'tasks/dense_ceval.json').read_text()),payload)
            self.assertTrue((out/'results.json').exists())

    def test_live_single_slot_measurement_is_not_whole_model_saving(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        torch.manual_seed(13);torch.set_num_threads(2)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=160,
            num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128)).eval()
        policy=study.SpatialConfig(representation='hierarchy',max_group=8,mask_layout='separate').pipeline(.4,k_layers=[0],v_layers=[])
        payload=evaluate_chunks(model,[torch.randint(0,32,(72,))],policy,[1],scoring_prefix=64)
        measured=study.slot_measurement(payload)
        self.assertAlmostEqual(measured['compression'],payload['storage']['targets']['k']['compression'])
        self.assertAlmostEqual(measured['compression'],2*payload['storage']['targets']['kv']['compression'])

if __name__=='__main__':unittest.main()
