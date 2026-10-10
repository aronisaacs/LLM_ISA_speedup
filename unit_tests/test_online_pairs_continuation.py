"""Live model reconstruction/accounting and causal continuation pilot protocol."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from compression_topics.spatial.algorithms import online_pairs as online
from compression_topics.spatial.scripts import online_pairs_continuation as study
from engine.eval_runner.chunks import evaluate_chunks
from engine.eval_runner.files import write_json
from engine.kv_compress import metrics

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(9)

    def test_pipeline_reconstruction_matches_proxy_decisions_and_accounting(self):
        x=torch.randn(1,2,65,128)
        table,recon=online.pair_table(x,target='v',return_reconstruction=True)
        expected=online.run(table,.25)
        metrics.reset_stats()
        y=online.apply(x,layer_idx=0,target='v',saving=.25)
        names=['dense','r0','r8','r16','r32']
        for n,entry in enumerate(expected['history']):
            i=names.index(entry['format'])
            torch.testing.assert_close(y[...,2*n:2*n+2,:],recon[i][...,n,:,:])
        self.assertTrue(torch.equal(y[...,-1,:],x[...,-1,:]))
        stats=metrics.STATS['v_layer_0']
        self.assertEqual(stats['stored_bits'],expected['stored_bits'])
        self.assertEqual(stats['dense_bits'],x.numel()*16)
        metrics.reset_stats()

    def test_live_full_model_three_arm_pilot_and_summary(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=128,intermediate_size=128,
            num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,
            max_position_embeddings=128)).eval()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            choices=[{'rung':2,'budget':.3,'assignment':[{'layer':0,'target':t,'budget':.25,'residual_entries':8} for t in ('k','v')]}]
            run=study.build_run(out,choices,budgets=(.3,),chunks=2,layers=1,prefix=32,suffix=8)
            self.assertEqual(len(run['configurations']),4)
            self.assertEqual(run['sampling']['split'],'validation')
            self.assertEqual(run['sampling']['scoring_prefix'],32)
            ids=[torch.arange(40)%64,(torch.arange(40)+5)%64]
            for config in run['configurations']:
                result=evaluate_chunks(model,ids,config['kv'],[1,2],scoring_prefix=32)
                result['simulation']={'kv':config['kv']}
                write_json(config['output_path'],result)
                self.assertEqual(sum(r['tokens'] for r in result['chunk_scores']),16)
                if config['name']!='dense':
                    self.assertEqual({s['method'] for s in config['kv']['pipeline']},{'online_pairs'})
                    self.assertNotIn('query_weights',config['kv']['pipeline'][0])
                    self.assertEqual(result['storage']['scope'],'simulated_prefill_cache_storage')
            report=study.summarize(out,run)
            self.assertEqual(report['status'],'complete')
            self.assertEqual({r['arm'] for r in report['rows']},set(study.ARMS))
            self.assertTrue(all(0<r['actual_prefill_kv_saving']<.5 for r in report['rows']))
            # Reject stale policy payloads rather than accepting an unrelated old run.
            config=run['configurations'][1];path=Path(config['output_path'])
            payload=json.loads(path.read_text());payload['simulation']['kv']={'pipeline':[]};write_json(path,payload)
            with self.assertRaisesRegex(ValueError,'policy differs'):study.summarize(out,run)

if __name__=='__main__':unittest.main()
