import unittest
import torch
from compression_topics.spatial.algorithms import online_pairs as online
from engine.kv_compress.rope import RopeTables,apply_rope

class OnlinePairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(41)

    def test_rope_alignment_and_head_local_reconstruction(self):
        rope=RopeTables(10000.,128)
        x=torch.randn(1,2,1,128).expand(1,2,64,128).clone()
        cos,sin=rope.cos_sin(torch.arange(64),torch.float32)
        rotated=apply_rope(x,cos,sin,inverse=False)
        a=online.pair_table(rotated,target='k',rope_tables=rope)
        b=online.pair_table(x,target='v')
        torch.testing.assert_close(torch.tensor(a.errors),torch.tensor(b.errors),atol=1e-5,rtol=1e-3)
        self.assertLess(max(e[1] for e in b.errors),1e-4)
        self.assertEqual(b.payload_bits[1],2*(16*(128+2)))
        with self.assertRaisesRegex(ValueError,'RoPE'):online.pair_table(x,target='k')

    def test_startup_and_budget_account_for_tags_and_dense_tail(self):
        x=torch.randn(1,2,129,128)
        table=online.pair_table(x,target='v')
        result=online.run(table,.3)
        self.assertLessEqual(result['startup_saving'],.3+1e-9)
        self.assertEqual(result['history'][15]['price_used'],result['initial_price'])
        self.assertEqual(result['stored_bits'],result['history'][-1]['stored_bits']+2*128*16)
        self.assertEqual(result['dense_bits'],x.numel()*16)
        self.assertEqual(sum(result['formats'].values()),64)
        self.assertEqual(result['history'][15]['recent_saving'],result['startup_saving'])

    def test_future_pairs_cannot_change_post_startup_decisions(self):
        x=torch.randn(1,2,128,128)
        a=online.pair_table(x,target='v')
        changed=x.clone();changed[...,96:,:]=torch.randn_like(changed[...,96:,:])*4
        b=online.pair_table(changed,target='v')
        ra,rb=online.run(a,.3),online.run(b,.3)
        self.assertEqual(ra['history'][:48],rb['history'][:48])

    def test_feedback_direction_and_bounds_on_changing_difficulty(self):
        # Warmup supports a .3 target. Later extreme errors forbid worthwhile merging,
        # so undercompression must drive the price upward; no future-price search occurs.
        errors=[[0.,.1,.06,.03,.01] for _ in range(16)]
        errors += [[0.,1e8,1e8,1e8,1e8] for _ in range(128)]
        table=online.PairTable(errors,[4096,2080,2336,2464,2720],4096,0)
        result=online.run(table,.3,control=online.Control(price_span=10))
        fixed=online.run(table,.3,feedback=False)
        self.assertGreater(result['history'][-1]['price_next'],result['initial_price'])
        self.assertLessEqual(result['history'][-1]['price_next'],result['initial_price']*10)
        self.assertGreater(result['price_bound_hits'],0)
        self.assertTrue(all(r['price_used']==fixed['initial_price'] for r in fixed['history']))
        self.assertEqual(online.outside(.01,.02),0.)
        self.assertAlmostEqual(online.outside(-.05,.02),-.03)

    def test_stationary_sequence_tracks_targets_after_startup(self):
        torch.manual_seed(44)
        table=online.pair_table(torch.randn(1,2,2048,128),target='v')
        for target in (.1,.25,.4,.49):
            result=online.run(table,target)
            self.assertLess(abs(result['measured_saving']-target),.03)
            self.assertGreater(result['fraction_post_startup_within_three_points'],.9)
            self.assertEqual(result['price_bound_hits'],0)

    def test_early_tolerance_shrinks_with_the_impact_of_a_single_decision(self):
        table=online.PairTable([], [4096,2080,2336,2464,2720],4096,0)
        control=online.Control()
        early=online.effective_bands(table,list(range(5)),16,control)
        later=online.effective_bands(table,list(range(5)),128,control)
        self.assertGreater(early[1],.06)
        self.assertLess(later[1],.034)
        self.assertAlmostEqual(early[2]/later[2],8.)
        # The same 4-point deviation is tolerated early but corrected later.
        self.assertEqual(online.outside(.04,early[1]),0.)
        self.assertGreater(online.outside(.04,later[1]),0.)
        self.assertEqual(online.asymmetric_outside(.04,early[1],control.cumulative_deadband),0.)
        self.assertLess(online.asymmetric_outside(-.04,early[1],control.cumulative_deadband),0.)

    def test_trace_exposes_early_actual_savings_and_effective_tolerances(self):
        table=online.pair_table(torch.randn(1,2,256,128),target='v')
        result=online.run(table,.3)
        self.assertEqual([r['tokens'] for r in result['early_checkpoints']], [32,48,64,96,128,192,256])
        bands=[r['cumulative_undercompression_band'] for r in result['early_checkpoints']]
        self.assertTrue(all(r['cumulative_overcompression_band']==.03 for r in result['early_checkpoints']))
        self.assertTrue(all(a>b for a,b in zip(bands,bands[1:])))
        self.assertEqual(result['early_checkpoints'][0]['stored_bits'],result['history'][15]['stored_bits'])
        self.assertGreaterEqual(result['fraction_post_startup_within_effective_band'],
                                result['fraction_post_startup_within_three_points'])

    def test_startup_prefers_less_compression_when_tied_pairs_jump_past_target(self):
        # Every pair has the same breakpoint: the only choices are near 0% or 49%.
        # A 30% startup target must prefer the dense side, rather than 49% savings.
        table=online.PairTable([[0.,.1,.1,.1,.1] for _ in range(16)],
                               [4096,2080,2336,2464,2720],4096,0)
        result=online.run(table,.3)
        self.assertLessEqual(result['startup_saving'],.3)
        self.assertEqual(result['formats']['dense'],16)
        self.assertGreater(result['initial_price'],0)

    def test_exact_startup_saving_does_not_back_off_unnecessarily(self):
        table=online.PairTable([[0.,.1,.1,.1,.1] for _ in range(16)],
                               [4096,2080,2336,2464,2720],4096,0)
        exact=1-(16*2080+online.metadata_bits(16,3))/(16*4096)
        result=online.run(table,exact)
        self.assertAlmostEqual(result['startup_saving'],exact)

    def test_pair_only_targets_and_invalid_configuration_rejected(self):
        table=online.pair_table(torch.randn(1,1,32,128),target='v')
        with self.assertRaises(ValueError):online.run(table,.5)
        with self.assertRaises(ValueError):online.Control(startup_tokens=31)
        with self.assertRaises(ValueError):online.Control(memory_pairs=0)
        with self.assertRaisesRegex(ValueError,'cannot meet'):online.run(table,.45,fixed_residual=32)

if __name__=='__main__':unittest.main()
