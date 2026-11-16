#!/usr/bin/env python3
"""Joint-KL push to 85 and 90 % mean saving on whole-bit levels (4 / 3 / 2 / 1), then the tasks.

Starts from the joint-KL 85 % map of the 4 / 3 / 2.5 / 2 study (``joint_kl/llama31``) with every
2.5-bit slot moved back to 3 bits (84.4 % saving), and continues the same climb: same calibration
set, score (4-domain KL to dense, divided by uniform 3 bits), lazy rounds and 32-token calibration
window. No split level: every slot is 4, 3, 2 or 1 bits (Aron, 2026-11-16).

Tasks (GSM8K, HumanEval, CEval, default window: 128 in job 1886, 32 since; per-question samples): the starting map, the
85 % and 90 % maps, and uniform 2 bits (87.5 %, the plain reference). Dense rows already in
results.json are reused.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_joint_kl_whole.py
  python compression_topics/turboquant_mse/scripts/turboquant_mse_joint_kl_whole.py --dry-run   # the start map only

The search resumes from ``state.json`` in ``joint_kl/llama31_whole``; task rows in results.json are skipped.
GPUs: one scorer or task worker per id in CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE, turboquant_mse  # noqa: E402
from compression_topics.turboquant_mse.scripts import turboquant_mse_joint_kl as jkl  # noqa: E402
from engine.layer_select.levels import mean_compression  # noqa: E402
from engine.layer_select.rungs import TURBOQUANT_MSE_WHOLE  # noqa: E402
from engine.layer_select.slots import Slot  # noqa: E402

BUDGETS = (0.85, 0.90)
TAG = "turboquant_mse_jklw_"
OUT = jkl.OUT.parent / "llama31_whole"
START_FROM = jkl.OUT / f"selected_{jkl.TAG}p85.json"


def start_map(path: Path = START_FROM) -> dict[Slot, float]:
    """The joint-KL 85 % map with 2.5-bit slots at 3 bits."""
    selection = json.loads(path.read_text())
    return {Slot.from_dict(a): (3 if a["level"] == 2.5 else a["level"]) for a in selection["assignment"]}


def start_selection(start: dict[Slot, float]) -> dict:
    n_slots = 2 * json.loads(START_FROM.read_text())["n_layers"]
    saving = mean_compression(start, n_slots, TURBOQUANT_MSE_WHOLE)
    selection = {"tag": f"{TAG}start", "budget": saving, "compression": saving,
                 "assignment": [{**s.to_dict(), "level": lvl} for s, lvl in sorted(start.items())]}
    return jkl.task_kv(selection)


def uniform_2_bits() -> dict:
    return {"tag": "turboquant_mse_u2", "kv": turboquant_mse(bits=2), "budget": 0.875, "compression": 0.875}


def saved_selections() -> list[dict]:
    paths = [OUT / f"selected_{TAG}p{round(b * 100):02d}.json" for b in BUDGETS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"search not finished, missing {missing} in {OUT}")
    return [jkl.task_kv(json.loads(p.read_text())) for p in paths]


def main() -> None:
    parser = jkl.parser()
    parser.description = __doc__
    parser.add_argument("--dry-run", action="store_true", help="print the start map and the task runs, run nothing")
    args = parser.parse_args()

    start = start_map()
    first = start_selection(start)
    if args.dry_run:
        counts = {lvl: sum(v == lvl for v in start.values()) for lvl in (4, 3, 2, 1)}
        print(f"start {START_FROM.name} with 2.5 -> 3: saving {first['compression']:.4f}, slots per level {counts}")
        return

    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table
    from engine.eval_runner.progress import mirror_terminal

    mirror_terminal(ROOT)
    ensure_table(TURBOQUANT_OUTLIER_TABLE)
    if not args.tasks_only:
        jkl.search(args, rungs=TURBOQUANT_MSE_WHOLE, budgets=BUDGETS, tag=TAG, out_dir=OUT, start=start,
                   extra={"start_from": f"{START_FROM.parent.name}/{START_FROM.name}, 2.5 -> 3"})
        if args.timing or args.search_only or args.model:
            return
    run = jkl.tasks_run([first, *saved_selections(), uniform_2_bits()])
    print(f"=== tasks: {len(run['configurations'])} runs ===", flush=True)
    jkl._multi_run(run)


if __name__ == "__main__":
    main()
