"""Tree optimality, causal windows, byte accounting and real cache integration."""
import itertools
import unittest
import torch
from compression_topics.spatial.algorithms import hierarchy
from compression_topics.spatial.algorithms.composed import SpatialConfig
from engine.kv_compress import metrics, parse_kv_spec
from engine.kv_compress.pipeline import compress_kv
from engine.kv_compress.rope import RopeTables, apply_rope


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31);torch.set_num_threads(2);metrics.reset_stats()
        self.rope = RopeTables(10000., 128)

    def tearDown(self):
        metrics.reset_stats()

    def test_tree_matches_exhaustive_small_window(self):
        # Exhaustive enumeration is only a test oracle, never the algorithm.
        x = torch.randn(1, 2, 8, 128)
        table = hierarchy.build_table(x,  head_decisions='shared', target='v')
        row = table.rows[0]
        def possibilities(start, size):
            options = [hierarchy.Choice(((start, 2, None),), 0., table.payload[(2, None)])] if size == 2 else [
                hierarchy.Choice(a.segments+b.segments, a.error+b.error, a.payload_bits+b.payload_bits)
                for a, b in itertools.product(possibilities(start, size//2), possibilities(start+size//2, size//2))]
            options += [hierarchy.Choice(((start, size, r),), e, table.payload[(size, r)])
                        for r, e in zip(hierarchy.MENUS[size], row[(size, start)])]
            return options
        options = possibilities(0, 8)
        for price in (0., .01, .5, 2., 100.):
            expected = min(o.error + price*o.bits/table.dense_window_bits for o in options)
            actual = table.choose(row, price)
            self.assertAlmostEqual(actual.error+price*actual.bits/table.dense_window_bits, expected)

    def test_local_choices_allow_different_children_and_use_all_heads(self):
        # A perfect first pair and unrelated second pair: only the first merges.
        a = torch.randn(1, 2, 1, 128)
        x = torch.cat((a, a, torch.randn(1, 2, 6, 128)), 2)
        table = hierarchy.build_table(x,  head_decisions='shared', target='v', max_group=2)
        choice = table.choose(table.rows[0], .01)
        self.assertEqual(choice.segments[0], (0, 2, 0))
        self.assertIn((2, 2, None), choice.segments)
        reverse = x.clone();reverse[:, :, :4] = x[:, :, [2, 3, 0, 1]]
        swapped = hierarchy.build_table(reverse,  head_decisions='shared', target='v', max_group=2)
        self.assertEqual(swapped.choose(swapped.rows[0], .01).segments[1], (2, 2, 0))
        damaged = x.clone();damaged[:, 1, 1] = torch.randn(1, 128)
        other = hierarchy.build_table(damaged,  head_decisions='shared', target='v', max_group=2)
        self.assertGreater(other.rows[0][(2, 0)][0], table.rows[0][(2, 0)][0])

    def test_rope_alignment_norms_tail_and_byte_accounting(self):
        base = torch.randn(1, 2, 1, 128)
        plain = base.expand(1, 2, 64, 128).clone()
        cos, sin = self.rope.cos_sin(torch.arange(64), torch.float32)
        keys = apply_rope(plain, cos, sin, inverse=False)
        keys = torch.cat((keys, torch.randn(1, 2, 1, 128)), 2).to(torch.bfloat16)
        table, representations = hierarchy.build_table(keys,  head_decisions='shared', target='k', rope_tables=self.rope,
                                                       return_reconstruction=True)
        result = hierarchy.run(table, .7)
        restored = hierarchy.restore(keys, result, representations)
        reference = keys.clone()
        for window,item in enumerate(result['history']):
            for start,size,residual in item['segments']:
                if residual is not None:
                    offset=window*8+start
                    reference[...,offset:offset+size,:]=representations[(size,residual)][...,offset:offset+size,:]
        self.assertTrue(torch.equal(restored,reference))
        self.assertTrue(torch.equal(restored[..., -1, :], keys[..., -1, :]))
        self.assertLess(float((restored.float()-keys.float()).square().mean()), 1e-3)
        choices = [table.choose(row, entry['price_used']) for row, entry in zip(table.rows, result['history'])]
        self.assertEqual(result['stored_bits'], hierarchy.packed_bits(choices)+table.tail_bits)
        self.assertEqual(result['dense_bits'], keys.numel()*16)
        self.assertEqual(1+sum(len(menu) for menu in hierarchy.MENUS.values()), 8)

    def test_later_windows_do_not_change_prior_decisions(self):
        x = torch.randn(1, 2, 128, 128)
        y = x.clone();y[:, :, 64:] *= torch.rand(1, 2, 64, 128)*10
        first = hierarchy.run(hierarchy.build_table(x,  head_decisions='shared', target='v'), .4)
        second = hierarchy.run(hierarchy.build_table(y,  head_decisions='shared', target='v'), .4)
        self.assertEqual(first['history'][:8], second['history'][:8])
        self.assertEqual(first['initial_price'], second['initial_price'])

    def test_shared_mask_uses_group_features_and_saves_only_mask_bytes(self):
        from compression_topics.spatial.algorithms.representations import prepare_groups, reconstruct
        from compression_topics.spatial.algorithms.storage import vector_group_bits
        x = torch.randn(1,2,32,128)
        for size in (2,4,8):
            basis = prepare_groups(x, target='v', group_size=size, directional=True, rope_tables=None)
            indices = basis.deviation.square().sum(-2, keepdim=True).topk(16, dim=-1).indices
            automatic, _ = reconstruct(basis,16,dtype=torch.bfloat16,shared_mask=True)
            explicit, _ = reconstruct(basis,16,dtype=torch.bfloat16,indices=indices,shared_mask=True)
            self.assertTrue(torch.equal(automatic,explicit))
            separate = vector_group_bits(16,128,size,include_norms=True)
            shared = vector_group_bits(16,128,size,include_norms=True,shared_mask=True)
            self.assertEqual(separate-shared, 0 if size==2 else (size-1)*128)
            self.assertEqual(vector_group_bits(0,128,size,include_norms=True),
                             vector_group_bits(0,128,size,include_norms=True,shared_mask=True))
        plain = hierarchy.build_table(x, head_decisions='shared', target='v',mask_layout='separate')
        shared = hierarchy.build_table(x, head_decisions='shared', target='v',mask_layout='shared')
        for row_a,row_b in zip(plain.rows,shared.rows):
            for start in (0,2,4,6):
                self.assertEqual(row_a[(2,start)],row_b[(2,start)])
        for count in (0,16,32):
            self.assertEqual(plain.payload[(2,count)],shared.payload[(2,count)])

    def test_engine_pipeline_dense_append_and_invalid_controls(self):
        x = torch.randn(1, 2, 128, 128).to(torch.bfloat16)
        spec = parse_kv_spec(SpatialConfig(representation='hierarchy', max_group=4).pipeline(.4))
        keys, values = compress_kv(x, x, 0, spec, rope=self.rope)
        self.assertTrue(torch.isfinite(keys).all() and torch.isfinite(values).all())
        # compress_kv does not call cache-level observe; kernel counters suffice here.
        self.assertIn('stored_bits', metrics.STATS['k_layer_0'])
        before = dict(metrics.STATS['k_layer_0'])
        appended, _ = compress_kv(x, x, 0, spec, seq_start=128, rope=self.rope)
        self.assertTrue(torch.equal(appended, x));self.assertEqual(before, metrics.STATS['k_layer_0'])
        for options in ({'decision':'offline'}, {'select_by':'query'}, {'control':{'startup_tokens':30}}):
            with self.assertRaises(ValueError):SpatialConfig(representation='hierarchy', **options)
        with self.assertRaisesRegex(ValueError, 'cannot be met'):
            hierarchy.run(hierarchy.build_table(x,  head_decisions='shared', target='v', max_group=2), .7)

    def test_live_prefill_compresses_both_slots_and_continuation_stays_dense(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        config = LlamaConfig(vocab_size=32, hidden_size=512, intermediate_size=640,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=128, max_position_embeddings=128)
        model = LlamaForCausalLM(config).eval()
        prompt = torch.randint(0,32,(72,))
        for layout in ('separate','shared'):
            policy = SpatialConfig(representation='hierarchy',mask_layout=layout).pipeline(.6)
            result = evaluate_chunks(model, [prompt], policy, [1], scoring_prefix=64)
            self.assertTrue(torch.isfinite(torch.tensor(result['results']['wikitext_chunks']['mean_nll,none'])))
            for target in ('k','v'):
                row = result['gate_stats'][f'{target}_layer_0']
                self.assertEqual(row['updates'],1)
                self.assertEqual(row['dense_bits'],64*2*128*16)
                self.assertLess(row['stored_bits'],row['dense_bits'])
                self.assertEqual(row['accounting'],hierarchy.accounting(layout))

if __name__ == '__main__':
    unittest.main()
