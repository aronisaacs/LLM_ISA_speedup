#!/usr/bin/env python3
"""Both TurboQuant allocators in one job, then one shared task run.

  A  rank fill on a per-slot WikiText perplexity sweep   (``turboquant_mse_study.py``)
  B  joint greedy on 4-domain KL to dense                 (``turboquant_mse_joint_kl.py``)

Steps: A's sweep, B's search, then dense and the six maps (A and B at 70/80/85 %) on GSM8K,
HumanEval and CEval in one queue, longest task first, so no GPU waits between the arms and
dense runs once. Every step resumes: the sweep and the tasks skip rows in results.json, the
search continues from its ``state.json``. Per-question scores (``.samples.jsonl``) are kept
for the paired A vs B comparison.

Window: both calibrations keep the newest 32 tokens exact (``TURBOQUANT_CALIBRATION_WINDOW``),
since they score prompt + answer in one pass; the task runs keep the usual 128.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_calibrate.py

GPUs: every id in CUDA_VISIBLE_DEVICES (one worker or scorer per GPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE  # noqa: E402
from compression_topics.turboquant_mse.scripts import turboquant_mse_joint_kl as arm_b  # noqa: E402
from compression_topics.turboquant_mse.scripts import turboquant_mse_study as arm_a  # noqa: E402


def tasks_run(selections: list[dict]) -> dict:
    """Dense once plus every map, each task's runs together, longest task first, dense last."""
    return arm_b.tasks_run(selections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-only", action="store_true", help="skip the sweep and the search (both finished)")
    parser.add_argument("--dry-run", action="store_true", help="print the task run instead of running it")
    args = parser.parse_args()

    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table
    from engine.eval_runner.progress import mirror_terminal

    mirror_terminal(ROOT)
    ensure_table(TURBOQUANT_OUTLIER_TABLE)
    if not args.tasks_only:
        print("=== A: per-slot WikiText sweep ===", flush=True)
        arm_a._multi_run(arm_a.sweep_run())
        print("=== B: joint-KL search ===", flush=True)
        arm_b.search(arm_b.parser().parse_args([]))
    run = tasks_run(arm_a.rank_fill_selections() + arm_b.saved_selections())
    if args.dry_run:
        for configuration in run["configurations"]:
            print(configuration["name"], "samples" if configuration.get("log_samples") else "")
        return
    print(f"=== tasks: {len(run['configurations'])} runs ===", flush=True)
    arm_b._multi_run(run)


if __name__ == "__main__":
    main()
