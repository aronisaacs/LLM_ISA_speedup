import unittest
import torch
from compression_topics.spatial.scripts.pair_diagnostic import apply_policy,attention_metrics,slot_policy

class DiagnosticTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(2);torch.set_num_threads(2)
    def test_matched_reference_and_reconstruction(self):
        x=torch.randn(1,2,128,64)
        a,am=apply_policy(x,'v',None,16,.2)
        b,bm=apply_policy(x,'v',None,None,am['actual_saving'],matched=True)
        self.assertLess(abs(bm['actual_saving']-am['actual_saving']),.01)
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        self.assertGreater(am['relative_reconstruction_error'],0)
    def test_attention_identity_and_causality(self):
        q=torch.randn(1,4,8,32);k=torch.randn(1,2,40,32);v=torch.randn_like(k)
        w=torch.eye(128)
        result=attention_metrics(q,k,v,k[:,:,:32],'k',32,w)
        self.assertEqual(result['projected_output_relative_error'],0)
        changed=v[:,:,:32].clone()+1
        result=attention_metrics(q,k,v,changed,'v',32,w)
        self.assertGreater(result['attention_output_relative_error'],0)
        self.assertEqual(result['logit_mse'],0)
    def test_live_query_capture_and_projected_error(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from compression_topics.spatial.scripts.pair_diagnostic import capture_states
        from engine.kv_compress.install import _rope_tables
        from types import SimpleNamespace
        import transformers.models.llama.modeling_llama as llama
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=160,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=64,max_position_embeddings=128)).eval()
        original=llama.apply_rotary_pos_emb
        states=capture_states(model,torch.arange(72)%32,(0,1),8)
        self.assertIs(llama.apply_rotary_pos_emb,original)
        q,k,v=states[1]
        self.assertEqual(tuple(q.shape),(1,2,8,64))
        changed,metrics=apply_policy(k[:,:,:64],'k',_rope_tables(SimpleNamespace(model=model)),16,.2)
        result=attention_metrics(q,k,v,changed,'k',64,model.model.layers[1].self_attn.o_proj.weight)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))
        self.assertGreater(result['projected_output_relative_error'],0)

    def test_dense_slot_policy(self):
        self.assertEqual(slot_policy({'kv':{'pipeline':[]}},0,'k'),(0.,None))
if __name__=='__main__':unittest.main()
