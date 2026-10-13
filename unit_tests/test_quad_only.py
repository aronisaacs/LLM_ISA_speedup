import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from compression_topics.spatial.algorithms import windows, methods
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import quad_only as study


class QuadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)

    def test_only_quads_or_dense_and_independent_heads(self):
        x = torch.randn(1, 2, 64, 64)
        x[:, 0] = x[:, 0, :1].expand(1, 64, 64)
        table, recon = windows.build(
            x, target="v", max_group=4, merge_sizes=(4,), residual_counts={"4": 16}
        )
        self.assertEqual(set(recon), {(4, 16)})
        choices = [table.choose(w, 0.01) for w in range(len(table))]
        self.assertTrue(
            all(
                r is None or (size == 4 and r == 16)
                for c in choices
                for h in c.heads
                for _, size, r in h.segments
            )
        )
        self.assertNotEqual(choices[0].heads[0].segments, choices[0].heads[1].segments)
        restored = windows.restore(x, choices, recon)
        self.assertTrue(torch.isfinite(restored).all())

    def test_configuration_and_candidates(self):
        p = study.plan(Path("/private/tmp/quad-plan"))
        rows = study.candidates(p)
        self.assertEqual(len(rows), 576)
        for r in rows:
            step = r["kv"]["pipeline"][0]
            self.assertEqual(step["merge_sizes"], [4])
            self.assertEqual(set(step["residual_counts"]), {"4"})
            self.assertTrue(step["prefill_only"])
        with self.assertRaises(ValueError):
            SpatialConfig(max_group=4, merge_sizes=(8,))

    def test_staged_validation_and_tasks(self):
        from unit_tests.test_spatial_studies import fake_chunks, fake_tasks

        with tempfile.TemporaryDirectory() as folder:
            p = study.plan(Path(folder), layers=1)
            p["local_budgets"] = [0.1, 0.4, 0.55]
            p["dense_ceval_source"] = str(Path(folder) / "missing.json")
            ref = SpatialConfig(prefill_only=True).pipeline(
                0.4, k_layers=[0], v_layers=[0]
            )
            with patch.object(
                study.shared, "reference_selection", return_value=ref
            ), patch.object(study.base, "execute_chunks", side_effect=fake_chunks):
                for stage in ("screen", "refine", "allocate", "validate"):
                    study.run_stage(p, stage)
            with patch.object(study.base, "execute_tasks", side_effect=fake_tasks):
                study.run_stage(p, "tasks")
            report = study.base.load(Path(folder) / "summary.json")
            self.assertEqual(report["completed"], 2)
            self.assertIn("reference_comparison", report)
            selections = study.base.load(Path(folder) / "selections.json")
            selections[0]["kv"]["pipeline"][0]["saving"] += 0.01
            from engine.eval_runner.files import write_json

            write_json(Path(folder) / "selections.json", selections)
            with patch.object(study.base, "execute_tasks") as execute:
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    study.run_stage(p, "tasks")
                execute.assert_not_called()

    def test_prefill_measurement_excludes_dense_continuation(self):
        payload = {
            "gate_stats": {
                "k_layer_0": {
                    "accounting": study.base.ACCOUNTING,
                    "dense_bits": 4096,
                    "stored_bits": 2867.2,
                }
            },
            "simulation": {
                "kv": {"pipeline": [{"method": "spatial_online", "prefill_only": True}]}
            },
            "scoring": {"prefix_tokens": 1536},
            "chunk_scores": [{"tokens": 512}],
            "storage": {"scope": "simulated_prefill_cache_storage"},
        }
        self.assertAlmostEqual(study.base.slot_measurement(payload)["compression"], 0.4)
        payload["simulation"]["kv"]["pipeline"][0]["prefill_only"] = False
        self.assertAlmostEqual(study.base.slot_measurement(payload)["compression"], 0.3)

    def test_live_query_and_prefill_only(self):
        import hashlib
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=160,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                max_position_embeddings=128,
            )
        ).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "q.pt"
            torch.save({"weights": torch.ones(1, 1, 64)}, path)
            config = SpatialConfig(
                max_group=4,
                merge_sizes=(4,),
                residual_counts={"4": 16},
                prefill_only=True,
                query_aware=True,
                query_weights=str(path),
                query_weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            result = evaluate_chunks(
                model,
                [torch.arange(72) % 32],
                config.pipeline(0.2, k_layers=[0], v_layers=[]),
                [0],
                scoring_prefix=64,
            )
            self.assertTrue(
                torch.isfinite(torch.tensor(result["chunk_scores"][0]["nll"]))
            )
            result["simulation"] = {
                "kv": config.pipeline(0.2, k_layers=[0], v_layers=[])
            }
            measured = study.base.slot_measurement(result)
            self.assertAlmostEqual(
                measured["compression"],
                result["storage"]["targets"]["k"]["compression"],
                places=6,
            )
            for stat in result["gate_stats"].values():
                self.assertEqual(stat.get("format_g2_r0", 0), 0)
                self.assertEqual(stat.get("format_g8_r0", 0), 0)
        methods.reset()


if __name__ == "__main__":
    unittest.main()
