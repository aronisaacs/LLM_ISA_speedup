"""Offline reference must share online reconstruction and exact byte accounting."""
import unittest
import torch
from compression_topics.spatial.algorithms import online_pairs as pairs
from engine.kv_compress import metrics

class OfflineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(27)
        metrics.reset_stats()

    def tearDown(self):
        metrics.reset_stats()

    def test_budget_ties_and_tail_preserve_one_pair_granularity(self):
        # All pairs break at the same price. Boundary repair must switch only
        # enough pairs instead of jumping the entire sequence to one format.
        table=pairs.PairTable([[0, .1, .1, .1, .1]]*100,[1000,520,600,700,800],1000,500)
        result=pairs.run_offline(table,.3)
        self.assertGreaterEqual(result['measured_saving'],.3)
        self.assertLess(result['measured_saving']-.3,480/100500)
        self.assertGreater(result['formats']['dense'],0)
        self.assertGreater(result['formats']['r0'],0)
        payload=sum(table.payload_bits[['dense','r0','r8','r16','r32'].index(r['format'])] for r in result['history'])
        self.assertEqual(result['stored_bits'],payload+pairs.metadata_bits(100,3)+500)
        with self.assertRaisesRegex(ValueError,'cannot be met'):
            pairs.run_offline(table,.49)

    def test_live_reconstruction_uses_identical_menu_and_statistics(self):
        x=torch.randn(1,2,65,128)
        table,options=pairs.pair_table(x,target='v',return_reconstruction=True)
        expected=pairs.run_offline(table,.3)
        y=pairs.apply(x,layer_idx=0,target='v',saving=.3,decision='offline')
        names=['dense','r0','r8','r16','r32']
        for n,entry in enumerate(expected['history']):
            torch.testing.assert_close(y[...,2*n:2*n+2,:],options[names.index(entry['format'])][...,n,:,:])
        self.assertTrue(torch.equal(y[...,-1,:],x[...,-1,:]))
        self.assertEqual(metrics.STATS['v_layer_0']['stored_bits'],expected['stored_bits'])
        self.assertTrue(torch.equal(pairs.apply(x,layer_idx=0,target='v',saving=.3,decision='offline',seq_start=65),x))
        with self.assertRaisesRegex(ValueError,'no fixed residual'):
            pairs.apply(x,layer_idx=0,target='v',saving=.3,decision='offline',fixed_residual=8)

    def test_full_model_keys_and_values_with_dense_continuation(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=128,intermediate_size=128,
            num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,
            max_position_embeddings=128)).eval()
        spec={'pipeline':[{'method':'online_pairs','decision':'offline','saving':.3,
                           'accounting':pairs.ACCOUNTING,'k_layers':'all','v_layers':'all'}]}
        result=evaluate_chunks(model,[torch.arange(72)%64],spec,[1],scoring_prefix=64)
        self.assertEqual(result['chunk_scores'][0]['tokens'],8)
        self.assertEqual(set(result['gate_stats']),{'k_layer_0','v_layer_0'})
        for row in result['gate_stats'].values():
            self.assertGreaterEqual(1-row['stored_bits']/row['dense_bits'],.3)
            self.assertEqual(row['updates'],1)

if __name__=='__main__':unittest.main()
