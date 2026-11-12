#!/usr/bin/env python3
"""TurboQuant per-slot bit widths by joint greedy on 4-domain KL, then CEval, GSM8K and HumanEval.

Same ladder (dense, 4, 3, 2.5, 2 bits), budgets (70/80/85 % mean saving) and tasks as
``turboquant_mse_study.py``, which allocates with a per-slot WikiText perplexity sweep and
rank fill. Only the allocator differs (``engine/layer_select/joint_kl``):

  score    KL(dense || compressed) on 512 calibration examples (WikiText, ARC 5-shot,
           GSM8K 5-shot, MBPP 3-shot; dense model's own answers), per domain divided by
           the KL of uniform 3 bits, equal-weight mean over the four domains
  search   from dense, each round scores every eligible slot's next rung on top of the
           current map and keeps the smallest kl_mean increase per unit of saving;
           lazy rounds score the 16 best estimates, every 5th round scores all slots

The search resumes from ``state.json`` in its output folder. Task rows already in
results.json are skipped.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_joint_kl.py            # search, then tasks
  python compression_topics/turboquant_mse/scripts/turboquant_mse_joint_kl.py --search-only
  python compression_topics/turboquant_mse/scripts/turboquant_mse_joint_kl.py --timing 16 # 16 ex/domain, 2 rounds

GPUs: one scorer per id in CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE as TABLE, turboquant_mse  # noqa: E402
from engine.layer_select.budgets import budget_run  # noqa: E402
from engine.layer_select.joint_kl import calibration_set  # noqa: E402
from engine.layer_select.joint_kl.search import SearchConfig, run_search  # noqa: E402
from engine.layer_select.rungs import TURBOQUANT_MSE  # noqa: E402
from engine.layer_select.slots import all_slots  # noqa: E402

BUDGETS = (0.70, 0.80, 0.85)
REFERENCE_BITS = 3
TAG = "turboquant_mse_jkl_"
OUT = ROOT / "compression_topics" / "turboquant_mse" / "joint_kl" / "llama31"
TASK_ORDER = ("gsm8k", "humaneval_instruct", "ceval")  # longest first


def devices() -> list[str]:
    import torch

    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["mps" if torch.backends.mps.is_available() else "cpu"]


def search(args) -> list[dict]:
    from transformers import AutoConfig

    from engine.layer_select.joint_kl.pool import ScorerPool

    rows, meta = calibration_set.load(per_domain=args.timing)
    model = args.model or meta["model"]
    n_layers = int(AutoConfig.from_pretrained(model).num_hidden_layers)
    slots = all_slots(n_layers)
    config = SearchConfig(
        method_kv=turboquant_mse(),
        rungs=TURBOQUANT_MSE,
        slots=slots,
        budgets=BUDGETS,
        reference={slot: REFERENCE_BITS for slot in slots},
        lazy_top=args.lazy_top,
        full_every=args.full_every,
        tag_prefix=TAG,
        max_rounds=2 if args.timing else None,
        extra={"model": model, "manifest_sha256": meta["manifest_sha256"], "examples": len(rows),
               "reference_map": f"uniform {REFERENCE_BITS} bits"},
    )
    out = Path(args.out) if args.out else (OUT.parent / f"timing_{args.timing}" if args.timing else OUT)
    found = devices()
    print(f"devices {found}, {len(rows)} examples, {len(slots)} slots -> {out}", flush=True)
    pool = ScorerPool(found, rows, meta, config.method_kv, model=args.model, dtype=args.dtype)
    try:
        dense = pool({"dense": {}})["dense"]  # the scorer compares dense with itself: must be 0
        if any(abs(v) > 1e-6 for v in dense.values()):
            raise RuntimeError(f"dense-vs-dense KL is not 0: {dense}")
        return run_search(pool, config, out, log=lambda *a: print(*a, flush=True))
    finally:
        pool.close()


def tasks_run(selections: list[dict]) -> dict:
    run = budget_run(selections, log_samples=True)  # per-question scores for the paired comparison
    task = lambda c: next(i for i, name in enumerate(TASK_ORDER) if f"llama31_{name}_" in c["name"])  # noqa: E731
    run["configurations"].sort(key=lambda c: (c["name"].endswith("_dense"), task(c)))
    return run


def saved_selections() -> list[dict]:
    paths = [OUT / f"selected_{TAG}p{round(b * 100):02d}.json" for b in BUDGETS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"search not finished, missing {missing} in {OUT}")
    return [json.loads(p.read_text()) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--tasks-only", action="store_true", help="task runs from a finished search")
    parser.add_argument("--timing", type=int, metavar="N", help="N examples per domain, 2 rounds, own folder")
    parser.add_argument("--lazy-top", type=int, default=16, help="0 = score every slot every round")
    parser.add_argument("--full-every", type=int, default=5)
    parser.add_argument("--out", help="output folder (default: joint_kl/llama31 or joint_kl/timing_N)")
    parser.add_argument("--model", help="checks only: another model with the same tokenizer")
    parser.add_argument("--dtype", help="checks only: e.g. float32 on a Mac")
    args = parser.parse_args()

    from engine.eval_runner.progress import mirror_terminal
    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table

    mirror_terminal(ROOT)
    ensure_table(TABLE)
    if not args.tasks_only:
        search(args)
        if args.timing or args.search_only or args.model:
            return
    _multi_run(tasks_run(saved_selections()))


def _multi_run(run: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"],
            cwd=ROOT, check=True, env=dict(os.environ),
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
