"""Sampling partitions, finalist diagnostics, and tiny-model chunk evaluation."""
import unittest
import torch
from compression_topics.spatial.scripts.pair_calibration_chunks import partition, evaluate_chunks, combine, worker_configurations
from compression_topics.spatial.scripts.pair_calibrated_study import comparisons, plan, subset_manifest


class SamplingTests(unittest.TestCase):
    def test_six_workers_cover_candidates_once_without_overlap(self):
        configs = list(range(17))
        partitions = [worker_configurations(configs, worker, 6) for worker in range(6)]
        self.assertEqual(sorted(sum(partitions, [])), configs)
        self.assertEqual(len(set(sum(partitions, []))), 17)
        self.assertEqual(partitions[0], [0, 6, 12])
        with self.assertRaises(ValueError):
            worker_configurations(configs, 6, 6)

    def test_extension_combines_losses_and_counts_without_recomputing_prefix(self):
        def result(chunk, nll):
            return {'chunk_scores': [{'chunk': chunk, 'nll': nll, 'tokens': 10}],
                    'gate_stats': {'k_layer_0': {'dense_bits': 100, 'stored_bits': 80,
                        'features': 128, 'kept': 16, 'group_size': 4,
                        'accounting': 'metadata_v1', 'cosine_min': .95, 'merged': 5}}}
        merged = combine(result(1, 1.), result(2, 3.))
        self.assertEqual(merged['results']['wikitext_chunks']['mean_nll,none'], 2.)
        self.assertEqual(merged['gate_stats']['k_layer_0']['dense_bits'], 200)
        self.assertEqual(merged['gate_stats']['k_layer_0']['kept'], 16)
        self.assertEqual(merged['gate_stats']['k_layer_0']['group_size'], 4)
        self.assertEqual(merged['gate_stats']['k_layer_0']['accounting'], 'metadata_v1')
        with self.assertRaises(ValueError):
            combine(result(1, 1.), result(1, 3.))

    def test_stages_are_reproducible_disjoint_and_nested(self):
        prompts, ids = list(range(80)), list(range(100, 180))
        base = {'seed': 3, 'offset': 0, 'chunks': 16}
        screen = partition(prompts, ids, base)[1]
        refine = partition(prompts, ids, {**base, 'offset': 16, 'chunks': 32})[1]
        extended = partition(prompts, ids, {**base, 'offset': 16, 'chunks': 64})[1]
        self.assertEqual(screen, partition(prompts, ids, base)[1])
        self.assertFalse(set(screen) & set(extended))
        self.assertEqual(refine, extended[:32])
        self.assertEqual(len(set(screen + extended)), 80)
        self.assertNotEqual(screen, sorted(screen))

    def test_refinement_contains_only_shortlisted_candidates(self):
        from pathlib import Path
        manifest = plan(Path('/tmp/sample'), layers=1)
        finalists = manifest['candidates'][:2]
        refined = subset_manifest(manifest, finalists, Path('/tmp/refined'), 16, 32)
        self.assertEqual(len(refined['candidates']), 2)
        self.assertEqual(len(refined['run']['configurations']), 3)
        self.assertEqual(refined['run']['sampling']['offset'], 16)
        self.assertEqual(refined['run']['sampling']['chunks'], 32)

    def test_paired_gap_flags_close_and_changed_orderings(self):
        def candidate(name, scores, ppl):
            return {'name': name, 'layer': 0, 'target': 'k', 'budget': .1, 'compression': .1,
                    'ppl': ppl, 'chunk_scores': [{'chunk': i, 'nll': v} for i, v in enumerate(scores)]}
        best = candidate('a', [1., 2., 3., 4.], 10.)
        other = candidate('b', [1.1, 1.9, 3.1, 3.9], 10.001)
        calibrated = {'selected': [best], 'candidates': [best, other], 'budget_tolerance': .001}
        screen = {'selected': [best]}
        self.assertTrue(comparisons(calibrated, screen)[0]['near_tie'])
        other['chunk_scores'] = [{'chunk': i, 'nll': v + .1} for i, v in enumerate([1., 2., 3., 4.])]
        self.assertFalse(comparisons(calibrated, screen)[0]['needs_more'])
        screen['selected'] = [other]
        self.assertTrue(comparisons(calibrated, screen)[0]['needs_more'])
        self.assertFalse(comparisons(calibrated, screen, trigger_changed=False)[0]['needs_more'])

    def test_random_tiny_model_runs_dense_and_compressed_chunks(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(5)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32,
            intermediate_size=48, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=64)).eval()
        prompts = [torch.randint(0, 32, (16,)), torch.randint(0, 32, (16,))]
        dense = evaluate_chunks(model, prompts, {'pipeline': []}, [5, 9])
        kv = {'pipeline': [{'method': 'pair_calibrated', 'k_layers': [0], 'v_layers': [],
                            'saving': .1, 'residual_entries': 0}]}
        compressed = evaluate_chunks(model, prompts, kv, [5, 9])
        self.assertEqual([r['chunk'] for r in compressed['chunk_scores']], [5, 9])
        self.assertEqual(sum(r['tokens'] for r in dense['chunk_scores']), 30)
        self.assertEqual(compressed['gate_stats']['k_layer_0']['updates'], 2)
        self.assertGreater(compressed['results']['wikitext_chunks']['token_perplexity,none'], 0)


if __name__ == '__main__':
    unittest.main()
