#!/usr/bin/env python3
"""TurboQuant task runs again with the 32-token exact window (the default since 2026-11-16).

Jobs 1864 and 1886 ran the tasks with the newest 128 tokens exact, which kept most of each
contract-task prompt exact. This reruns the same maps on GSM8K, HumanEval and CEval with the
default window (32), under tags ending in ``_w32`` so the 128-window files stay as they are:

  whole bits (job 1886)   start map, joint KL 85 % and 90 % (4 / 3 / 2 / 1), uniform 2 bits
  --ab (job 1864)         rank fill and joint KL at 70 / 80 / 85 % (4 / 3 / 2.5 / 2)

Per-question samples are kept for paired tests. Dense rows already in results.json are skipped.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_tasks_w32.py
  python compression_topics/turboquant_mse/scripts/turboquant_mse_tasks_w32.py --ab
  python compression_topics/turboquant_mse/scripts/turboquant_mse_tasks_w32.py --dry-run

GPUs: one task worker per id in CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE, TURBOQUANT_RECENT_WINDOW  # noqa: E402
from compression_topics.turboquant_mse.scripts import turboquant_mse_joint_kl as jkl  # noqa: E402
from compression_topics.turboquant_mse.scripts import turboquant_mse_joint_kl_whole as whole  # noqa: E402

WINDOW = 32
SUFFIX = f"_w{WINDOW}"


def selections(ab: bool = False) -> list[dict]:
    chosen = [whole.start_selection(whole.start_map()), *whole.saved_selections(), whole.uniform_2_bits()]
    if ab:
        from compression_topics.turboquant_mse.scripts import turboquant_mse_study as arm_a

        chosen += [*arm_a.rank_fill_selections(), *jkl.saved_selections()]
    for selection in chosen:
        windows = {step.get("recent_window") for step in selection["kv"]["pipeline"]}
        if windows != {WINDOW}:
            raise SystemExit(f"{selection['tag']}: window {windows}, expected {WINDOW}")
    return [{**selection, "tag": selection["tag"] + SUFFIX} for selection in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ab", action="store_true", help="also the six maps of the rank fill vs joint KL job")
    parser.add_argument("--dry-run", action="store_true", help="print the runs, run nothing")
    args = parser.parse_args()
    if TURBOQUANT_RECENT_WINDOW != WINDOW:
        raise SystemExit(f"default window is {TURBOQUANT_RECENT_WINDOW}, expected {WINDOW}")

    run = jkl.tasks_run(selections(args.ab))
    if args.dry_run:
        for configuration in run["configurations"]:
            print(configuration["name"])
        print(f"{len(run['configurations'])} runs (dense ones are skipped if already in results.json)")
        return

    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table
    from engine.eval_runner.progress import mirror_terminal

    mirror_terminal(ROOT)
    ensure_table(TURBOQUANT_OUTLIER_TABLE)
    print(f"=== tasks, window {WINDOW}: {len(run['configurations'])} runs ===", flush=True)
    jkl._multi_run(run)


if __name__ == "__main__":
    main()
