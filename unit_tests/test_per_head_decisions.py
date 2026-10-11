"""Head-local decisions, aggregate controller, causal state, and exact storage."""
import unittest
import torch
from compression_topics.spatial.algorithms import hierarchy,online_pairs as pairs
from compression_topics.spatial.algorithms.decisions import run_head_pairs
from compression_topics.spatial.algorithms.storage import metadata_bits
from engine.kv_compress import metrics

class HeadDecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14);torch.set_num_threads(2);metrics.reset_stats()
    def tearDown(self):metrics.reset_stats()
    def source(self,length=64):
        x=torch.randn(1,2,length,128)
        x[:,0,1::2]=x[:,0,::2]
        return x
    def test_pair_heads_choose_different_formats_at_same_price(self):
        x=self.source();table=pairs.build_pair_table(x,target='v',head_decisions='per_head')
        from compression_topics.spatial.algorithms.decisions import select
        rates=[b/table.dense_pair_bits for b in table.payload_bits]
        chosen=[select(row,rates,.01,range(5)) for row in table.errors[0]]
        self.assertEqual(chosen,[1,0])
        result=run_head_pairs(table,.2)
        self.assertTrue(any(len(set(r['formats']))>1 for r in result['history']))
        out=pairs.apply(x,layer_idx=0,target='v',saving=.2)
        stat=metrics.STATS['v_layer_0']
        self.assertEqual(stat['pairs'],64)
        self.assertEqual(stat['metadata_bits'],metadata_bits(64,3))
        self.assertEqual(stat['stored_bits'],result['stored_bits'])
        self.assertLessEqual(result['startup_saving'],.2+1e-9)
        for i,r in enumerate(result['history']):
            for h,format in enumerate(r['formats']):
                if format=='dense':self.assertTrue(torch.equal(out[:,h,2*i:2*i+2],x[:,h,2*i:2*i+2]))
    def test_hierarchy_heads_choose_and_restore_independently(self):
        x=self.source();table,recon=hierarchy.build_table(x,target='v',head_decisions='per_head',return_reconstruction=True)
        c=table.choose(table.rows[0],.01)
        self.assertEqual(c.head_segments[0][0],(0,2,0))
        self.assertEqual(c.head_segments[1][0],(0,2,None))
        result=hierarchy.run(table,.2);out=hierarchy.restore(x,result,recon)
        self.assertEqual(result['dense_bits'],x.numel()*16)
        count=sum(len(segments) for r in result['history'] for segments in r['head_segments'])
        self.assertEqual(result['metadata_bits'],metadata_bits(count,3))
        reference=x.clone()
        payload=0
        for w,r in enumerate(result['history']):
            for h,segments in enumerate(r['head_segments']):
                for start,size,residual in segments:
                    payload+=table.tables[h].payload[(size,residual)]
                    if residual is not None:
                        off=w*8+start;reference[:,h,off:off+size]=recon[(size,residual)][:,h,off:off+size]
        self.assertTrue(torch.equal(out,reference))
        self.assertEqual(result['stored_bits'],payload+metadata_bits(count,3))
        self.assertLessEqual(result['startup_saving'],.2+1e-9)
    def test_future_heads_or_tokens_do_not_change_previous_decisions(self):
        x=self.source(128);changed=x.clone();changed[:,:,96:]=torch.randn_like(changed[:,:,96:])
        for build,run in [(lambda x: pairs.build_pair_table(x,target='v',head_decisions='per_head'),run_head_pairs),
                          (lambda x: hierarchy.build_table(x,target='v',head_decisions='per_head'),hierarchy.run)]:
            a=run(build(x),.3);b=run(build(changed),.3)
            before=lambda r:[i for i in r['history'] if i['tokens']<=96]
            self.assertEqual(before(a),before(b))
    def test_head_permutation_preserves_controller_and_reconstruction(self):
        x=self.source()
        for apply in (pairs.apply,hierarchy.apply):
            metrics.reset_stats();a=apply(x,layer_idx=0,target='v',saving=.3);stats=dict(metrics.STATS['v_layer_0'])
            metrics.reset_stats();b=apply(x[:,[1,0]],layer_idx=0,target='v',saving=.3)
            self.assertTrue(torch.equal(a,b[:,[1,0]]))
            self.assertEqual(stats['stored_bits'],metrics.STATS['v_layer_0']['stored_bits'])
    def test_single_head_matches_shared_reference_and_shared_masks_work(self):
        x=torch.randn(1,1,64,128)
        for method in (pairs.apply,hierarchy.apply):
            metrics.reset_stats();a=method(x,layer_idx=0,target='v',saving=.3,head_decisions='per_head')
            bits=metrics.STATS['v_layer_0']['stored_bits']
            metrics.reset_stats();b=method(x,layer_idx=0,target='v',saving=.3,head_decisions='shared')
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(bits,metrics.STATS['v_layer_0']['stored_bits'])
        metrics.reset_stats()
        x=self.source();table,recon=hierarchy.build_table(x,target='v',mask_layout='shared',return_reconstruction=True)
        result=hierarchy.run(table,.4)
        out=hierarchy.restore(x,result,recon)
        actual=hierarchy.apply(x,layer_idx=0,target='v',saving=.4,mask_layout='shared')
        self.assertTrue(torch.equal(out,actual))
        self.assertEqual(result['stored_bits'],metrics.STATS['v_layer_0']['stored_bits'])
        self.assertTrue(any(r['head_segments'][0]!=r['head_segments'][1] for r in result['history']))

    def test_invalid_head_does_not_prevent_other_head_merge(self):
        x=self.source(8);x[:,1,:2]=0
        t=hierarchy.build_table(x,target='v',head_decisions='per_head')
        c=t.choose(t.rows[0],.01)
        self.assertEqual(c.head_segments[0][0],(0,2,0))
        self.assertEqual(c.head_segments[1][0],(0,2,None))

if __name__=='__main__':unittest.main()
