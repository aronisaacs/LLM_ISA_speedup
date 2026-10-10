"""Composition must preserve legacy methods and independently control scoring."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from compression_topics.spatial.algorithms import composed, online_pairs, presentation_spatial
from engine.kv_compress import metrics, parse_kv_spec
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables

class SpatialComponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17);metrics.reset_stats()
        self.x=torch.randn(1,2,65,128).to(torch.bfloat16)
        self.rope=RopeTables(10000.,128)

    def tearDown(self):metrics.reset_stats()

    def test_compositions_match_legacy_outputs_and_bytes(self):
        for target in ('k','v'):
            for decision in ('online','offline'):
                metrics.reset_stats()
                expected=online_pairs.apply(self.x,layer_idx=0,target=target,saving=.3,decision=decision,rope_tables=self.rope)
                bits=metrics.STATS[f'{target}_layer_0']['stored_bits']
                metrics.reset_stats()
                actual=composed.apply(self.x,layer_idx=0,target=target,saving=.3,decision=decision,rope_tables=self.rope)
                self.assertTrue(torch.equal(expected,actual));self.assertEqual(metrics.STATS[f'{target}_layer_0']['stored_bits'],bits)
            for size in (2,4):
                metrics.reset_stats()
                expected=presentation_spatial.apply(self.x,layer_idx=0,target=target,saving=.2,group_size=size,residual_entries=8,rope_tables=self.rope)
                bits=metrics.STATS[f'{target}_layer_0']['stored_bits']
                metrics.reset_stats()
                actual=composed.apply(self.x,layer_idx=0,target=target,saving=.2,decision='fixed',representation='pairs' if size==2 else 'quads',residual_entries=8,rope_tables=self.rope)
                self.assertTrue(torch.equal(expected,actual));self.assertEqual(metrics.STATS[f'{target}_layer_0']['stored_bits'],bits)

    def test_distortion_feature_selection_and_importance_are_independent(self):
        basis,plain=online_pairs.build_pair_table(self.x,target='v',return_reconstruction=True)
        energy=torch.ones(1,2,1,1,128);energy[...,0]=1000.
        objective,unchanged=online_pairs.build_pair_table(self.x,target='v',return_reconstruction=True,distortion='query',query_energy=energy)
        features,changed=online_pairs.build_pair_table(self.x,target='v',return_reconstruction=True,selection='query',query_energy=energy)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(plain,unchanged)))
        self.assertNotEqual(basis.errors,objective.errors)
        self.assertFalse(torch.equal(plain[2],changed[2]))
        weighted=online_pairs.build_pair_table(self.x,target='v',token_weights=torch.full(self.x.shape[:3],2.))
        torch.testing.assert_close(torch.tensor(weighted.errors),2*torch.tensor(basis.errors))
        self.assertEqual(weighted.payload_bits,basis.payload_bits)

    def test_invalid_or_noncausal_combinations_fail_explicitly(self):
        for options,reason in [({'decision':'online','representation':'quads'},'quad'),
                               ({'decision':'online','importance':'prefill_attention'},'not causal'),
                               ({'select_by':'query'},'checksum'),
                               ({'decision':'fixed'},'residual count'),
                               ({'decision':'offline','control':{}},'controller'),
                               ({'residual_menu':(0,4,8)},'menu')]:
            with self.assertRaisesRegex(ValueError,reason):composed.SpatialConfig(**options)
            with self.assertRaisesRegex(ValueError,reason):
                parse_kv_spec({'pipeline':[{'method':'spatial_composed','saving':.3,**options}]})

    def test_pipeline_queries_and_offline_importance_reach_shared_components(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt';torch.save({'weights':torch.rand(1,2,128)+.1},path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            config=composed.SpatialConfig(decision='offline',select_by='query',distortion='query',
                importance='prefill_attention',query_weights=str(path),query_weights_sha256=digest)
            with self.assertRaisesRegex(ValueError,'values separately'):config.pipeline(.3)
            spec=parse_kv_spec(config.pipeline(.3,k_layers=[0],v_layers=[]))
            keys,values=compress_kv(self.x,self.x,0,spec,rope=self.rope,token_weights=torch.ones(self.x.shape[:3]))
            self.assertFalse(torch.equal(keys,self.x));self.assertTrue(torch.equal(values,self.x))
            self.assertGreaterEqual(1-metrics.STATS['k_layer_0']['stored_bits']/metrics.STATS['k_layer_0']['dense_bits'],.3)
            with self.assertRaisesRegex(ValueError,'requires matching'):
                compress_kv(self.x,self.x,0,spec,rope=self.rope)
            appended,_=compress_kv(self.x,self.x,0,spec,seq_start=65,rope=self.rope)
            self.assertTrue(torch.equal(appended,self.x))

    def test_live_model_attention_capture_and_query_calibration_compositions(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=128,intermediate_size=128,
            num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,
            max_position_embeddings=128)).eval()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt';torch.save({'weights':torch.rand(1,1,128)+.1},path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for decision,importance in (('online','none'),('offline','prefill_attention')):
                keys=composed.SpatialConfig(decision=decision,select_by='query',distortion='query',
                    importance=importance,query_weights=str(path),query_weights_sha256=digest)
                values=composed.SpatialConfig(decision=decision,importance=importance)
                spec={'pipeline':keys.pipeline(.3,k_layers=[0],v_layers=[])['pipeline']+
                                 values.pipeline(.3,k_layers=[],v_layers=[0])['pipeline']}
                result=evaluate_chunks(model,[torch.arange(72)%64],spec,[1],scoring_prefix=64)
                self.assertEqual(result['chunk_scores'][0]['tokens'],8)
                self.assertEqual(set(result['gate_stats']),{'k_layer_0','v_layer_0'})
                self.assertEqual(result['gate_stats']['k_layer_0']['updates'],1)
                self.assertTrue(torch.isfinite(torch.tensor(result['chunk_scores'][0]['nll'])))

if __name__=='__main__':unittest.main()
