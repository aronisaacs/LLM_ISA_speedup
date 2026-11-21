"""Spatial compression on left-padded batches: each row is compressed as in a batch of one."""

import unittest
from types import SimpleNamespace

import torch

from compression_topics.spatial.algorithms.methods import SpatialConfig
from engine.kv_compress import metrics, padding
from engine.kv_compress.install import install
from engine.kv_compress.spec import parse_kv_spec


class LeftPadTests(unittest.TestCase):
    def tearDown(self):
        padding.clear()

    def test_no_mask_is_fine_for_one_row_and_refused_for_more(self):
        self.assertEqual(padding.left_pads(1, 10), [0])
        with self.assertRaises(ValueError):
            padding.left_pads(2, 10)

    def test_pads_from_the_mask(self):
        padding.record(None, (), {"attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])})
        self.assertEqual(padding.left_pads(2, 4), [2, 0])

    def test_right_padding_and_shape_mismatch_are_refused(self):
        padding.record(None, (), {"attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])})
        with self.assertRaises(ValueError):
            padding.left_pads(2, 4)
        with self.assertRaises(ValueError):
            padding.left_pads(2, 5)


class BatchedGenerationTests(unittest.TestCase):
    def _model(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                                            max_position_embeddings=256)).eval()

    def _generate(self, model, ids, mask, kv):
        uninstall = install(SimpleNamespace(model=model), parse_kv_spec(kv))
        metrics.reset_stats()
        try:
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=40, min_new_tokens=40,
                                     do_sample=False, pad_token_id=0, eos_token_id=None)
        finally:
            uninstall()
        return out, metrics.pop_stats()

    def test_batched_rows_match_one_by_one(self):
        model = self._model()
        kv = SpatialConfig(window=32).pipeline(0.4, k_layers=[0, 1], v_layers=[0, 1])
        g = torch.Generator().manual_seed(1)
        prompts = [torch.randint(1, 64, (n,), generator=g) for n in (45, 70, 19)]
        width = max(len(p) for p in prompts)
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            ids[i, width - len(p):] = p
            mask[i, width - len(p):] = 1
        batched, stats = self._generate(model, ids, mask, kv)
        totals = {}
        for i, p in enumerate(prompts):
            solo, solo_stats = self._generate(model, p.unsqueeze(0), torch.ones(1, len(p), dtype=torch.long), kv)
            self.assertTrue(torch.equal(solo[0, len(p):], batched[i, width:]))
            for key, row in solo_stats.items():
                for field in ("dense_bits", "stored_bits", "metadata_bits"):
                    totals[(key, field)] = totals.get((key, field), 0) + row[field]
        for (key, field), value in totals.items():
            self.assertEqual(stats[key][field], value)
        dense, _ = self._generate(model, ids, mask, {"pipeline": []})
        self.assertFalse(torch.equal(dense, batched))  # the compression does act


if __name__ == "__main__":
    unittest.main()
