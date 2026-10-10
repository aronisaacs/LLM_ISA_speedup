"""Causality, query-selected features, shared storage, and study identities."""
import tempfile
from pathlib import Path
import unittest
import torch
from engine.kv_compress.importance import next8_attention
from engine.kv_compress.metrics import STATS
from compression_topics.spatial.algorithms import hierarchy
from compression_topics.spatial.algorithms.composed import SpatialConfig
from compression_topics.spatial.scripts import hierarchy_weighted as study

class WeightedTests(unittest.TestCase):
    def test_local_attention_matches_direct_causal_reference_and_is_bounded(self):
        torch.manual_seed(4)
        q=torch.randn(1,4,40,32);k=torch.randn(1,2,40,32)
        got=next8_attention(q,k)
        for start in range(0,32,8):
            scores=q[0,:,start+8:start+16]@k[0,:,start:start+16].repeat_interleave(2,0).transpose(-1,-2)/32**.5
            scores.masked_fill_(torch.arange(16)[None,:]>torch.arange(8,16)[:,None],float('-inf'))
            expected=(16*scores.softmax(-1)[...,:8].reshape(2,2,8,8).mean((1,2))).clamp_min(.1)
            torch.testing.assert_close(got[0,:,start:start+8],expected)
        q[...,16:,:]*=100;k[...,16:,:]*=100
        torch.testing.assert_close(got[...,:8],next8_attention(q,k)[...,:8])
        self.assertTrue((got[...,-8:]==1).all())

    def test_query_feature_selection_and_shared_bytes(self):
        torch.manual_seed(2);x=torch.randn(1,2,40,64)
        energy=torch.ones(1,2,1,1,64);energy[...,:16]=1000
        plain=hierarchy.build_table(x,target='v')
        query=hierarchy.build_table(x,target='v',selection='query',query_energy=energy)
        self.assertNotEqual(plain.rows,query.rows)
        shared=hierarchy.build_table(x,target='v',selection='query',query_energy=energy,mask_layout='shared')
        self.assertLess(shared.payload[(8,16)],query.payload[(8,16)])
        self.assertEqual(shared.payload[(2,16)],query.payload[(2,16)])

    def test_dense_tail_and_actual_bytes(self):
        torch.manual_seed(5);x=torch.randn(1,2,40,64);STATS.clear()
        out=hierarchy.apply(x,layer_idx=0,target='v',saving=.4,delayed=True,token_weights=torch.ones(1,2,40))
        self.assertTrue(torch.equal(out[...,-8:,:],x[...,-8:,:]))
        table=hierarchy.build_table(x[...,:32,:],target='v',token_weights=torch.ones(1,2,40))
        result=hierarchy.run(table,.4)
        row=STATS['v_layer_0'];tail=8*2*64*16
        self.assertEqual(row['stored_bits'],result['stored_bits']+tail)
        self.assertEqual(row['dense_bits'],x.numel()*16)
        STATS.clear()

    def test_live_query_capture_with_bounded_importance(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=160,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
            max_position_embeddings=128)).eval()
        with tempfile.TemporaryDirectory() as folder:
            plan=study.plan(Path(folder));plan['layers']=1
            torch.save({'weights':torch.ones(1,1,64)},plan['query_weights'])
            rows=study.candidates(plan)
            for arm in study.ARMS:
                steps=[]
                for target in ('k','v'):
                    steps+=next(r['kv']['pipeline'] for r in rows if r['arm']==arm and r['target']==target and r['budget']==.4)
                result=evaluate_chunks(model,[torch.arange(72)%32],{'pipeline':steps},[0],scoring_prefix=64)
                self.assertEqual(set(result['gate_stats']),{'k_layer_0','v_layer_0'})
                self.assertTrue(torch.isfinite(torch.tensor(result['chunk_scores'][0]['nll'])))
                for row in result['gate_stats'].values():self.assertEqual(row['updates'],1)

    def test_staged_calibration_keeps_four_arms_separate(self):
        from unittest.mock import patch
        from unit_tests.test_hierarchy_accuracy import HierarchyAccuracyTests
        import json
        fixture=HierarchyAccuracyTests()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);plan=study.plan(out,budgets=(.1,.4));plan['layers']=1
            plan['local_budgets']=[.1,.4,.85]
            torch.save({'weights':torch.ones(1,1,64)},plan['query_weights'])
            with patch.object(study.base.workflow,'execute_chunks',side_effect=fixture.fake_chunks):
                for stage in ('screen','refine','allocate','validate'):
                    study.base.workflow.run_stage(plan,stage,candidate_builder=study.candidates,arms=study.ARMS,
                        measure_fn=study.measure,allocation_fn=study.allocations,task_builder=study.task_run,summarizer=study.summarize)
            selections=json.loads((out/'selections.json').read_text())
            self.assertEqual(len(selections),8)
            self.assertEqual({s['arm'] for s in selections},set(study.ARMS))
            plan['dense_ceval_source']=str(out/'missing.json')
            tasks=study.task_run(plan,selections)
            self.assertEqual({c['metadata']['arm'] for c in tasks['configurations'][1:]},set(study.ARMS))

    def test_all_four_arms_have_distinct_weighted_policies(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=study.plan(Path(folder));plan['layers']=1
            torch.save({'weights':torch.ones(1,2,64)},plan['query_weights'])
            rows=study.candidates(plan)
            self.assertEqual(len(rows),56)
            self.assertEqual({r['arm'] for r in rows},set(study.ARMS))
            for row in rows:
                step=row['kv']['pipeline'][0]
                self.assertEqual(step['select_by'],'query' if row['target']=='k' else 'deviation')
                self.assertEqual(step['importance'],'next8_attention' if 'attention' in row['arm'] else 'none')
                self.assertEqual(step['mask_layout'],'shared' if 'shared' in row['arm'] else 'separate')
                SpatialConfig(**{k:v for k,v in step.items() if k in SpatialConfig.__dataclass_fields__})

if __name__=='__main__':unittest.main()
