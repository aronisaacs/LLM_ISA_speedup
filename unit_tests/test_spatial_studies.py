"""The staged ladder study: candidates, calibration, allocation, gates and tasks."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from compression_topics.spatial.algorithms.formats import ACCOUNTING
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import ladder, study
from engine.eval_runner.cache import simulation_identity
from engine.eval_runner.chunks import identity
from engine.eval_runner.files import write_json
from engine.kv_compress import parse_kv_spec


def fake_chunks(path, *, endpoint_saving=None):
    """Pretend each slot met its target; pair slots at 50% reach ``endpoint_saving``."""
    run = json.loads(Path(path).read_text())
    sampling = run["sampling"]
    for config in run["configurations"]:
        steps = config["kv"]["pipeline"]
        stats, slots, saving = {}, {}, 0.0
        for step in steps:
            endpoint = step["max_group"] == 2 and step["saving"] == 0.5
            achieved = endpoint_saving if endpoint and endpoint_saving else step["saving"]
            for target in ("k", "v"):
                for layer in step[target + "_layers"]:
                    key = f"{target}_layer_{layer}"
                    bits = {"dense_bits": 100000, "stored_bits": 100000 * (1 - achieved)}
                    slots[key] = bits
                    stats[key] = dict(bits, accounting=ACCOUNTING, updates=sampling["chunks"])
                    saving += achieved / (2 * run["model_shape"]["layers"])
        nll = 1.0 + 0.01 * saving
        payload = {
            "simulation": identity(config, sampling, run),
            "results": {"wikitext_chunks": {"token_perplexity,none": math.exp(nll)}},
            "chunk_scores": [
                dict(chunk=i, nll=nll, tokens=sampling["seq_len"] - sampling["scoring_prefix"])
                for i in range(sampling["offset"], sampling["offset"] + sampling["chunks"])
            ],
            "scoring": {"prefix_tokens": sampling["scoring_prefix"]},
            "gate_stats": stats,
        }
        if steps:
            payload["storage"] = {
                "scope": "simulated_prefill_cache_storage",
                "slots": slots,
                "targets": {"kv": {"compression": saving}},
            }
        write_json(config["output_path"], payload)


def fake_tasks(destination, run, *, summarizer):
    for config in run["configurations"]:
        _, key, metric, full = next(t for t in study.TASKS.values() if t[0]["tasks"] == config["tasks"])
        payload = {
            "simulation": simulation_identity(run, config, parse_kv_spec(config["kv"])),
            "n-samples": {key: {"effective": full}},
            "results": {key: {metric: 0.5}},
            "gate_stats": {},
        }
        if config["kv"]["pipeline"]:
            payload["storage"] = {
                "targets": {"kv": {"compression": config["metadata"]["global_budget"]}}
            }
        write_json(config["output_path"], payload)
    summarizer(destination, run, require_complete=True)


class LadderTests(unittest.TestCase):
    def test_candidates_per_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            p = ladder.plan(Path(folder))
            torch.save({"weights": torch.ones(32, 8, 128)}, p["query_weights"])
            rows = ladder.candidates(p)
            self.assertEqual(len(rows), 64 * (6 + 9 + 6 + 9))
            for row in rows:
                step = row["kv"]["pipeline"][0]
                self.assertEqual(step["method"], "spatial")
                self.assertEqual(step["max_group"], 2 if row["arm"] in ladder.PAIR_ARMS else 8)
                self.assertEqual(
                    step["query_aware"],
                    row["arm"] in ladder.QUERY_ARMS and row["target"] == "k",
                )
                self.assertEqual(
                    row["maximum_feasible"], row["arm"] in ladder.PAIR_ARMS and row["budget"] == 0.5
                )

    def test_simplification_arms_at_30_and_40(self):
        with tempfile.TemporaryDirectory() as folder:
            p = ladder.plan(Path(folder), arms=tuple(ladder.SIMPLIFICATIONS), budgets=(0.3, 0.4))
            torch.save({"weights": torch.ones(32, 8, 128)}, p["query_weights"])
            rows = ladder.candidates(p)
            self.assertEqual(len(rows), len(ladder.SIMPLIFICATIONS) * 64 * 5)  # no 50% endpoint
            self.assertFalse(any(r["maximum_feasible"] for r in rows))
            flags = {r["arm"]: r["kv"]["pipeline"][0] for r in rows if r["target"] == "k"}
            self.assertTrue(flags["query_pairs_per_head"]["per_head_budget"])
            self.assertFalse(flags["query_pairs_no_rope"]["align_rope"])
            self.assertFalse(flags["query_pairs_no_norms"]["keep_norms"])
            both = flags["query_pairs_no_rope_no_norms"]
            self.assertFalse(both["align_rope"] or both["keep_norms"])
            w16 = flags["query_pairs_per_head_w16"]
            self.assertEqual((w16["window"], w16["per_head_budget"]), (16, True))
            self.assertEqual(flags["query_pairs_w16"]["window"], 16)
            self.assertNotIn("per_head_budget", flags["query_pairs_w16"])
            padded = flags["query_pairs_per_head_padded_w16"]
            self.assertEqual((padded["window"], padded["per_head_budget"], padded["pad_heads"]),
                             (16, True, True))

    def test_head_split_arm_reads_per_head_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            p = ladder.plan(Path(folder), arms=(ladder.HEAD_SPLIT,), budgets=(0.3, 0.4), layers=2)
            torch.save({"weights": torch.ones(2, 8, 128)}, p["query_weights"])
            with self.assertRaises(ValueError):
                ladder.candidates(p)  # needs the head calibration first
            budgets = ladder.head_split_budgets(p)
            self.assertNotIn(0.5, budgets)
            shares = {str(l): {t: {str(b): [b] * 8 for b in budgets} for t in "kv"} for l in range(2)}
            levels = {str(l): {t: {str(b): [.25] * 8 for b in budgets} for t in "kv"} for l in range(2)}
            write_json(p["head_savings"], {"savings": shares, "levels": levels})
            rows = ladder.candidates(p)
            self.assertEqual(len(rows), 2 * 2 * len(budgets))
            for row in rows:
                self.assertEqual(row["kv"]["pipeline"][0]["head_savings"], [row["budget"]] * 8)
                self.assertNotIn("pad_heads", row["kv"]["pipeline"][0])
            p["arms"] = [ladder.HEAD_LEVELS]
            for row in ladder.candidates(p):
                step = row["kv"]["pipeline"][0]
                self.assertEqual((step["head_savings"], step["pad_heads"]), ([.25] * 8, True))

    def test_validation_gate_and_endpoint(self):
        rows = [
            dict(
                arm=a, budget=b, actual_saving=0.491 if a in ladder.PAIR_ARMS and b == 0.5 else b
            )
            for a in ladder.ARMS
            for b in ladder.BUDGETS
        ]
        ladder.check_validation(rows, ladder.BUDGETS, ladder.ARMS, tolerance=0.01)
        rows[0]["actual_saving"] = 0.25
        with self.assertRaises(ValueError):
            ladder.check_validation(rows, ladder.BUDGETS, ladder.ARMS, tolerance=0.01)

    def test_staged_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = ladder.plan(out, layers=1, screen=2, refine=2, validation=2)
            p["pair_local_budgets"] = [0.3, 0.4, 0.49]
            p["local_budgets"] = [0.3, 0.4, 0.5, 0.85]
            p["dense_ceval_source"] = str(out / "missing.json")
            torch.save({"weights": torch.ones(1, 2, 128)}, p["query_weights"])
            with patch.object(
                study,
                "execute_chunks",
                side_effect=lambda path: fake_chunks(path, endpoint_saving=0.491),
            ):
                for stage in ("screen", "refine", "allocate", "validate"):
                    ladder.run_stage(p, stage)
            selections = json.loads((out / "selections.json").read_text())
            self.assertEqual(len(selections), 12)
            endpoint = next(s for s in selections if s["maximum_feasible"])
            self.assertAlmostEqual(endpoint["compression"], 0.491)
            for selection in selections:
                self.assertTrue(
                    all(
                        row["arm"] == selection["arm"]
                        for row in selection["assignment"]
                    )
                )
            with patch.object(study, "execute_tasks", side_effect=fake_tasks):
                ladder.run_stage(p, "tasks")
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual(report["completed"], 13)
            self.assertEqual(
                sum(r.get("maximum_feasible", False) for r in report["rows"]), 2
            )
            validation = json.loads((out / "validation.json").read_text())
            validation[1]["actual_saving"] = 0.2
            write_json(out / "validation.json", validation)
            with patch.object(study, "execute_tasks") as execute:
                with self.assertRaisesRegex(
                    RuntimeError, "(byte gate|provenance check) failed"
                ):
                    ladder.run_stage(p, "tasks")
                execute.assert_not_called()


class OfficialSweepTests(unittest.TestCase):
    def test_three_tasks_per_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = ladder.plan(out, arms=(ladder.OFFICIAL_ARM,), budgets=ladder.OFFICIAL_BUDGETS,
                            layers=1, tasks=ladder.OFFICIAL_TASKS)
            p["dense_ceval_source"] = str(out / "missing.json")
            selections = [dict(arm=ladder.OFFICIAL_ARM, budget=b, tag=f"{ladder.OFFICIAL_ARM}_b{round(100 * b):02d}",
                               kv={"pipeline": []}, maximum_feasible=False) for b in ladder.OFFICIAL_BUDGETS]
            run = ladder.tasks(p, selections)
            names = [c["name"] for c in run["configurations"]]
            self.assertEqual(len(names), 3 * (1 + len(ladder.OFFICIAL_BUDGETS)))
            self.assertTrue(names[0].startswith("dense_gsm8k"))  # slow generation runs dealt first
            self.assertEqual({c["metadata"].get("operating_point") for c in run["configurations"]
                              if c["name"].startswith(ladder.OFFICIAL_ARM)}, {"calibrated_budget"})
            fake_tasks(out, run, summarizer=lambda *a, **k: None)
            report = ladder.summarize(out, run, require_complete=True)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(report["dense_accuracy_by_task"]), {"ceval", "gsm8k", "humaneval"})
            code = ladder.plan(out / "code", arms=(ladder.OFFICIAL_ARM,), budgets=ladder.OFFICIAL_BUDGETS,
                               layers=1, tasks=ladder.MODELS["codellama"][1],
                               model_args=ladder.MODELS["codellama"][0])
            qwen = ladder.plan(out / "qwen", arms=(ladder.OFFICIAL_ARM,), budgets=(0.3,),
                               model_args=ladder.MODELS["qwen"][0], **ladder.MODELS["qwen"][2])
            Path(qwen["out"]).mkdir()
            torch.save({"weights": torch.ones(32, 4, 256)}, qwen["query_weights"])
            self.assertEqual({r["layer"] for r in ladder.candidates(qwen)}, set(range(3, 32, 4)))
            code_run = ladder.tasks(code, selections)
            self.assertEqual({c["tasks"][0] for c in code_run["configurations"]}, {"humaneval_codellama"})
            self.assertIn("CodeLlama", code_run["model_args"])
            step = SpatialConfig(max_group=2, **ladder.SIMPLIFICATIONS[ladder.OFFICIAL_ARM])
            self.assertEqual((step.window, step.per_head_budget, step.pad_heads), (16, True, True))


class StudyTests(unittest.TestCase):
    def test_chunk_splits(self):
        p = ladder.plan(Path("/tmp/plan"))
        screen = study.chunk_run(p, [], "screen")["sampling"]
        refine = study.chunk_run(p, [], "refine")["sampling"]
        self.assertEqual((screen["split"], refine["split"]), ("train", "train"))
        self.assertEqual(refine["offset"], screen["chunks"])
        self.assertEqual(
            study.chunk_run(p, [], "validation")["sampling"]["split"], "validation"
        )

    def test_dense_reuse_requires_identity_and_full_question_count(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = ladder.plan(out, layers=1)
            source = out / "source.json"
            p["dense_ceval_source"] = str(source)
            config = study.configuration(
                "dense_ceval", {"pipeline": []}, out / "tasks", study.CEVAL_VALID_5SHOT
            )
            base = {"model": "hf", "model_args": p["model_args"], "batch_size": 1}
            payload = {
                "simulation": simulation_identity(
                    base, config, parse_kv_spec(config["kv"])
                ),
                "n-samples": {"ceval-valid": {"effective": 1346}},
                "results": {"ceval-valid": {"acc,none": 0.5}},
            }
            stale = deepcopy(payload)
            stale["simulation"]["num_fewshot"] = 0
            write_json(source, stale)
            study.task_run(p, [])
            self.assertFalse((out / "tasks/dense_ceval.json").exists())
            write_json(source, payload)
            study.task_run(p, [])
            self.assertEqual(
                json.loads((out / "tasks/dense_ceval.json").read_text()), payload
            )

    def test_live_single_slot_measurement(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        from compression_topics.spatial.algorithms.methods import SpatialConfig

        torch.manual_seed(13)
        torch.set_num_threads(2)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=160,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=128,
            )
        ).eval()
        policy = SpatialConfig().pipeline(0.4, k_layers=[0], v_layers=[])
        payload = evaluate_chunks(
            model, [torch.randint(0, 32, (72,))], policy, [1], scoring_prefix=64
        )
        measured = study.slot_measurement(payload)
        self.assertGreaterEqual(measured["compression"], 0.4)


if __name__ == "__main__":
    unittest.main()
