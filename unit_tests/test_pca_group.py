"""PCA accuracy identities, frozen-head policies, accounting and tiny-model flow."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from compression_topics.spatial.algorithms import pca_group as pca, pair_gate
from compression_topics.spatial.scripts.pca_group_study import choose_setting, pipeline, calibrate
from compression_topics.spatial.scripts.pair_calibration_chunks import evaluate_chunks, combine
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec


class PCATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        pair_gate.reset_stats()
        pca.SAMPLES.clear()

    def test_rank_one_sparse_coefficients_are_refitted(self):
        x = torch.nn.functional.normalize(torch.randn(5, 4, 128), dim=-1)
        residual = x - x.mean(-2, keepdim=True)
        _, _, vh = torch.linalg.svd(residual)
        direction = pca.half(pair_gate._largest(vh[:, 0], 64))
        coeff = (residual * direction[:, None]).sum(-1) / direction.square().sum(-1, keepdim=True)
        first = pca.half(coeff[:, :3])
        coeff = torch.cat((first, -first.sum(-1, keepdim=True)), -1)
        expected = torch.nn.functional.normalize(pca.half(x.mean(-2, keepdim=True)) + coeff[..., None] * direction[:, None], dim=-1)
        actual, errors = pca.reconstruct(x, 'rank1', 64)
        # Independent SVD/eigh fits can straddle fp16 rounding boundaries.
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)
        self.assertTrue(torch.isfinite(errors).all())

    def test_storage_counts_masks_norms_coefficients_basis_and_flags(self):
        merged, header, basis = pca.storage('rank1', 64)
        self.assertEqual(merged // 8, 414)
        self.assertEqual(header // 8, 5)
        self.assertEqual(basis, 0)
        self.assertEqual(pca.storage('rank1', 128)[0] // 8, 526)
        self.assertEqual(pca.storage('shared', 16)[2] // 8, 4096)
        self.assertEqual(pca.overhead('rank1', 64, 9), 40 + 16)

    def test_fixed_head_threshold_tail_and_decode(self):
        x = torch.randn(1, 2, 11, 128)
        settings = [{'kind': 'rank1', 'size': 64, 'threshold': 2.}, None]
        actual = pca.apply(x, layer_idx=0, target='v', head_settings=settings)
        self.assertTrue(torch.equal(actual[:, 1], x[:, 1]))
        self.assertTrue(torch.equal(actual[..., 8:, :], x[..., 8:, :]))
        stats = pair_gate.pop_stats()['v_layer_0_head_0']
        longer = torch.cat((x[..., :8, :], torch.randn(1, 2, 8, 128)), -2)
        extended = pca.apply(longer, layer_idx=0, target='v', head_settings=settings)
        torch.testing.assert_close(extended[..., :8, :], actual[..., :8, :])
        pair_gate.reset_stats()
        self.assertEqual(stats['merged'], 2)
        self.assertEqual(stats['stored_bits'], 2*pca.storage('rank1', 64)[0] + 3*128*16 + pca.overhead('rank1', 64, 2))
        self.assertTrue(torch.equal(pca.apply(x, layer_idx=0, target='v', head_settings=settings, seq_start=11), x))
        # Dense rejection uses the same fixed threshold independent of prompt length.
        threshold = [{'kind': 'rank1', 'size': 32, 'threshold': 0.}, None]
        self.assertTrue(torch.equal(pca.apply(x, layer_idx=0, target='v', head_settings=threshold), x))

    def test_shared_basis_storage_checksum_and_key_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'basis.pt'
            basis = torch.eye(128)[:, :16]
            torch.save({'k_layer_0': basis[None].half()}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Equal canonical token directions should survive mean-only correction.
            raw = torch.randn(1, 1, 1, 128).expand(1, 1, 8, 128).clone()
            rope = RopeTables(10000., 128)
            cos, sin = rope.cos_sin(torch.arange(8), torch.float32)
            keys = apply_rope(raw, cos, sin, inverse=False)
            settings = [{'kind': 'shared', 'size': 16, 'threshold': .01}]
            actual = pca.apply(keys, layer_idx=0, target='k', head_settings=settings,
                rope_tables=rope, basis_path=str(path), basis_digest=digest)
            torch.testing.assert_close(actual, keys, atol=.003, rtol=.002)
            stats = pair_gate.pop_stats()['k_layer_0_head_0']
            self.assertEqual(stats['basis_bits'], 128*16*16)
            self.assertEqual(stats['stored_bits'], 2*pca.storage('shared', 16)[0] + pca.overhead('shared', 16, 2))
            with self.assertRaises(ValueError):
                pca.load_bases(path, 'wrong')

    def test_calibration_one_setting_and_fixed_threshold(self):
        unit = torch.nn.functional.normalize(torch.randn(32, 4, 128), dim=-1)
        for kind, sizes in [('rank1', (32, 64, 128)), ('separate', (0, 8, 16))]:
            choice = choose_setting(unit, kind, sizes, .2, 2048)
            self.assertGreaterEqual(choice['calibration_saving'], .2)
            self.assertEqual(choice['threshold'], float(torch.tensor(choice['threshold']).half()))
            self.assertIn(choice['size'], sizes)

    def test_calibration_artifacts_and_tiny_model_evaluation(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            data = {slot: {'fit': torch.nn.functional.normalize(torch.randn(2, 8, 4, 16), dim=-1).half(),
                           'calibration': torch.nn.functional.normalize(torch.randn(2, 8, 4, 16), dim=-1).half()}
                    for slot in ('k_layer_0', 'v_layer_0')}
            torch.save(data, out / 'samples.pt')
            # Coarse grids use 128-D production heads; test smaller ranks explicitly.
            from compression_topics.spatial.scripts import pca_group_study as study
            original = study.SIZES
            study.SIZES = {'rank1': (4, 8, 16), 'shared': (2, 4), 'separate': (0, 2)}
            try:
                calibrate(out, SimpleNamespace(options=['rank1', 'shared'], budgets=[.1], seq_len=2048,
                    fit_chunks=1, calibration_chunks=1, screen_chunks=1, refine_chunks=1, pool_chunks=4, seed=0))
            finally:
                study.SIZES = original
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()
            policies = json.loads((out / 'policies.json').read_text())
            prompts = [torch.randint(0, 32, (16,))]
            for policy in policies:
                parse_kv_spec(policy['kv'])
                result = evaluate_chunks(model, prompts, policy['kv'], [1])
                self.assertTrue(math.isfinite(result['results']['wikitext_chunks']['token_perplexity,none']))
                self.assertEqual(len(result['gate_stats']), 4)
                extra = evaluate_chunks(model, prompts, policy['kv'], [2])
                combined = combine(result, extra)
                self.assertEqual(combined['gate_stats']['k_layer_0_head_0']['group_size'], 4)
                self.assertEqual(combined['gate_stats']['k_layer_0_head_0']['accounting'], pca.ACCOUNTING)
            collected = {'pipeline': [{'method': 'pca_quad_collect', 'k_layers': [0], 'v_layers': [0], 'samples_per_chunk': 2}]}
            evaluate_chunks(model, prompts, collected, [1])
            self.assertEqual(pca.SAMPLES['k_layer_0'][0].shape, (2, 2, 4, 16))


if __name__ == '__main__':
    unittest.main()
