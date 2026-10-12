import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from compression_topics.spatial.algorithms import hierarchy
from compression_topics.spatial.algorithms.composed import SpatialConfig
from compression_topics.spatial.scripts import online_fixed_rungs as study

class FixedRungTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(4);torch.set_num_threads(2)
    def test_fixed_format_choices_and_causality(self):
        x=torch.randn(1,2,128,128)
        for residual in (0,16,32):
            table,recon=hierarchy.build_table(x,target='v',max_group=2,residual_entries=residual,return_reconstruction=True)
            a=hierarchy.run(table,.2)
            for row in a['history']:
                for head in row['head_segments']:
                    self.assertTrue(all(size==2 and r in (None,residual) for _,size,r in head))
            output=hierarchy.restore(x,a,recon)
            self.assertTrue(torch.isfinite(output).all())
            changed=x.clone();changed[:,:,96:]*=20
            b=hierarchy.run(hierarchy.build_table(changed,target='v',max_group=2,residual_entries=residual),.2)
            self.assertEqual(a['history'][:12],b['history'][:12])
    def test_invalid_fixed_online_combinations(self):
        for options in [dict(max_group=8,residual_entries=16),dict(max_group=2,residual_entries=True),dict(max_group=2,residual_entries=8)]:
            with self.assertRaises(ValueError):SpatialConfig(representation='hierarchy',**options)
    def test_calibration_and_task_metadata(self):
        from unit_tests.test_hierarchy_accuracy import HierarchyAccuracyTests
        with tempfile.TemporaryDirectory() as folder:
            p=study.plan(Path(folder),layers=1);p['dense_ceval_source']=str(Path(folder)/'missing.json')
            rows=study.candidates(p)
            self.assertTrue(all(r['residual_entries']==0 for r in rows if r['arm']=='no_residual'))
            self.assertEqual({r['residual_entries'] for r in rows if r['arm']=='fixed_residual'},{0,16,32})
            with patch.object(study.base.workflow,'execute_chunks',side_effect=HierarchyAccuracyTests().fake_chunks):
                for stage in ('screen','refine','allocate','validate'):study.run_stage(p,stage)
            selections=json.loads((Path(folder)/'selections.json').read_text())
            self.assertEqual(len(selections),8)
            for s in selections:
                self.assertEqual(len(s['kv']['pipeline']),2)
                self.assertTrue(all(step['residual_entries'] in (0,16,32) for step in s['kv']['pipeline']))
            tasks=study.tasks(p,selections)
            self.assertEqual(len(tasks['configurations']),9)
            self.assertEqual({r['metadata']['arm'] for r in tasks['configurations'][1:]},set(study.ARMS))
    def test_live_rope_key_and_value_paths(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=160,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=64,max_position_embeddings=128)).eval()
        for residual in (0,16):
            kv=SpatialConfig(representation='hierarchy',max_group=2,residual_entries=residual).pipeline(.2)
            got=evaluate_chunks(model,[torch.arange(72)%32],kv,[0],scoring_prefix=64)
            self.assertTrue(torch.isfinite(torch.tensor(got['chunk_scores'][0]['nll'])))
            self.assertEqual(set(got['gate_stats']),{'k_layer_0','v_layer_0'})
if __name__=='__main__':unittest.main()
