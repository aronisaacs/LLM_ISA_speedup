"""Real model checks for prefix vs total cache, historical reads, and extension."""

from copy import deepcopy
import unittest
import torch
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts.study import slot_measurement
from engine.eval_runner.chunks import evaluate_chunks, combine
from engine.eval_runner.storage import sum_storage


class StorageContractTests(unittest.TestCase):
    def setUp(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(11)
        torch.set_num_threads(2)
        self.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=160,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                max_position_embeddings=128,
            )
        ).eval()

    def run_chunk(self, kv, lengths=(72, 80), ids=(0, 1)):
        result = evaluate_chunks(
            self.model,
            [torch.arange(n) % 32 for n in lengths],
            kv,
            list(ids),
            scoring_prefix=64,
        )
        result["simulation"] = {"kv": kv}
        return result

    def test_variable_continuations_have_explicit_prefix_bytes(self):
        kv = SpatialConfig(max_group=4).pipeline(0.2, k_layers=[0], v_layers=[])
        r = self.run_chunk(kv)
        slot = r["storage"]["slots"]["k_layer_0"]
        self.assertEqual(slot["dense_bits"], 2 * 64 * 64 * 16)
        self.assertEqual(
            r["cache_storage"]["slots"]["k_layer_0"]["dense_bits"], (72 + 80) * 64 * 16
        )
        m = slot_measurement(r)
        self.assertEqual(m["measurement_source"], "explicit_storage_slots")
        self.assertAlmostEqual(
            m["compression"], r["storage"]["targets"]["k"]["compression"] * 2
        )
        self.assertGreater(  # continuation windows are compressed too
            r["cache_storage"]["slots"]["k_layer_0"]["dense_bits"]
            - r["cache_storage"]["slots"]["k_layer_0"]["stored_bits"],
            slot["dense_bits"] - slot["stored_bits"],
        )

    def test_dense_and_decode_use_same_scope_contract(self):
        dense = self.run_chunk({"pipeline": []})
        self.assertEqual(len(dense["storage"]["slots"]), 4)
        self.assertEqual(dense["storage"]["targets"]["kv"]["compression"], 0)
        streaming = self.run_chunk(
            SpatialConfig(max_group=4).pipeline(0.2, k_layers=[0], v_layers=[])
        )
        self.assertEqual(
            streaming["storage"]["slots"]["k_layer_0"]["dense_bits"], 2 * 64 * 64 * 16
        )
        self.assertGreater(
            streaming["cache_storage"]["slots"]["k_layer_0"]["dense_bits"],
            streaming["storage"]["slots"]["k_layer_0"]["dense_bits"],
        )

    def test_extension_keeps_storage_and_maximum_diagnostics(self):
        kv = SpatialConfig(max_group=2).pipeline(
            0.2, k_layers=[0], v_layers=[]
        )
        a = self.run_chunk(kv, lengths=(72,), ids=(1,))
        b = self.run_chunk(kv, lengths=(80,), ids=(2,))
        a["gate_stats"]["k_layer_0"]["max_final_target_deviation"] = 0.1
        b["gate_stats"]["k_layer_0"]["max_final_target_deviation"] = 0.2
        r = combine(a, b)
        self.assertEqual(
            r["gate_stats"]["k_layer_0"]["max_final_target_deviation"], 0.2
        )
        self.assertEqual(
            r["storage"]["slots"]["k_layer_0"]["dense_bits"], 2 * 64 * 64 * 16
        )
        self.assertIn("scoring", r)
        self.assertIn("cache_storage", r)
        bad = deepcopy(b)
        bad["scoring"]["prefix_tokens"] = 32
        with self.assertRaises(ValueError):
            combine(a, bad)
        with self.assertRaises(ValueError):
            sum_storage([a["storage"], b["cache_storage"]])

    def test_saturation_stream_header_is_counted(self):
        from unit_tests.test_spatial_algorithms import cache_run, correlated
        from compression_topics.spatial.algorithms.formats import HEADER_BITS

        _, r = cache_run(correlated(64), [32, 32], saving=0.5, max_group=2)
        count = r["format_g2_r0"]
        self.assertEqual(r["metadata_bits"], HEADER_BITS + count * 3)


if __name__ == "__main__":
    unittest.main()
