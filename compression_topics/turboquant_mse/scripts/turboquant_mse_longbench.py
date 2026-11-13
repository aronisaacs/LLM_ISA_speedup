#!/usr/bin/env python3
"""TurboQuant on LongBench-E (Llama-3.1-8B-Instruct), set up as close to the paper's Table 1 as we can.

Not a contract task: an internal check against the paper's only Llama-3.1-8B numbers.

Truncated set (the current run; prompts cut to 31,500 tokens in the middle, as LongBench's pred.py does):
  dense
  paper     32 outlier channels at 3 bits + 96 at 2 (2.25 bits), as faithful to the paper as we can:
            a dense random orthogonal rotation per part, Lloyd-Max levels for the exact unit-vector
            coordinate distribution, the prompt attending to its own K/V uncompressed, the newest
            128 tokens exact (the QJL code by the paper's first author does both)
  out32_dprompt   the same split with our Hadamard + RoPE-pair signs and Gaussian levels
                  (shows whether the paper-faithful details matter)
  out64     64 + 64 (2.5 bits) with our defaults, including the 128-token window (our setup from now on)

Untruncated set (first run, DGX jobs 1805 and the 2.25-bit dense-prompt follow-up): dense, 32 + 96,
64 + 64, and both splits with dense prompt attention, no window; ``--summary`` still prints it.

The paper does not say how it picks outliers; ours come from the calibration table
(mean |value| per channel). Task setup: see LONGBENCH_E_* in catalog/tasks.py.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_longbench.py
  python compression_topics/turboquant_mse/scripts/turboquant_mse_longbench.py --limit 5   # timing run
  python compression_topics/turboquant_mse/scripts/turboquant_mse_longbench.py --summary

The timing run takes the first ``--limit`` examples of each task, for dense and out64 only,
and writes to logs/longbench_timing (results.json is not touched).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE as TABLE, turboquant_mse  # noqa: E402
from catalog.tasks import (  # noqa: E402
    LONGBENCH_E_CATEGORIES,
    LONGBENCH_E_CHAT,
    LONGBENCH_E_PLAIN,
    LONGBENCH_E_TR_CATEGORIES,
    LONGBENCH_E_TR_CHAT,
    LONGBENCH_E_TR_PLAIN,
)
from engine.layer_select.budgets import budget_run  # noqa: E402

HEAD_DIM = 128
TIMING_ROOT = "logs/longbench_timing"

# Paper, Table 1, Llama-3.1-8B-Instruct (scores x 100). Its "2.5" row is the 32 + 96 split (2.25 bits).
PAPER = {
    "dense": (45.29, 45.16, 26.55, 68.38, 59.54, 46.28, 50.06),
    "2.25 bits (32 + 96)": (44.16, 44.96, 24.80, 68.01, 59.65, 45.76, 49.44),
    "3.5 bits": (45.01, 45.31, 26.00, 68.63, 59.95, 46.17, 50.06),
}


def _split(outliers: int, dense_prompt: bool = False, window: int = 0, **options) -> dict:
    kv = turboquant_mse(bits=2, outliers=outliers, outlier_table=TABLE, recent_window=window, **options)
    if dense_prompt:
        kv = {**kv, "prompt_attention": "dense"}
    return kv


def _rows(rows) -> list[dict]:
    """Tag, kv and index-bit compression vs 16-bit values for each compressed variant."""
    out = []
    for tag, kv, outliers in rows:
        bits = (outliers * 3 + (HEAD_DIM - outliers) * 2) / HEAD_DIM
        out.append({"tag": f"turboquant_mse_{tag}", "kv": kv, "budget": 1 - bits / 16, "compression": 1 - bits / 16})
    return out


def selections() -> list[dict]:
    """The truncated run."""
    return _rows([
        ("paper_out32", _split(32, dense_prompt=True, window=128, rotation="orthogonal", codebook="beta"), 32),
        ("out32_3b_2b_dprompt_w128", _split(32, dense_prompt=True, window=128), 32),
        ("out64_3b_2b_w128", _split(64, window=128), 64),
    ])


def untruncated_selections() -> list[dict]:
    """The first, untruncated run (no window)."""
    return _rows([
        ("out32_3b_2b", _split(32), 32),
        ("out64_3b_2b", _split(64), 64),
        ("out64_3b_2b_dprompt", _split(64, dense_prompt=True), 64),
        ("out32_3b_2b_dprompt", _split(32, dense_prompt=True), 32),
    ])


def longbench_run(limit: int | None = None) -> dict:
    chosen = selections()
    tasks = (LONGBENCH_E_TR_CHAT, LONGBENCH_E_TR_PLAIN)
    if limit is None:
        run = budget_run(chosen, tasks=tasks)
    else:
        out64 = [c for c in chosen if c["tag"] == "turboquant_mse_out64_3b_2b_w128"]
        run = budget_run(out64, tasks=tasks, results_dir=TIMING_ROOT)
        for configuration in run["configurations"]:
            configuration["limit"] = limit
    # Longest first: the chat half (summaries up to 512 new tokens), then the plain half.
    order = ("longbench_e_tr_chat", "longbench_e_tr_plain")
    rank = lambda c: next(i for i, name in enumerate(order) if f"llama31_{name}_" in c["name"])
    run["configurations"].sort(key=rank)
    return run


def summary(results_path: Path) -> str:
    """Category means and the 13-task mean (x 100), as the paper's Table 1 reports them, for both task sets."""
    simulations = json.loads(results_path.read_text())["simulations"]
    header = ["single QA", "multi QA", "summ.", "few-shot", "synth.", "code", "average"]
    lines = [f"{'':40s}" + "".join(f"{h:>10s}" for h in header)]
    for name, values in PAPER.items():
        lines.append(f"{'paper ' + name:40s}" + "".join(f"{v:10.2f}" for v in values))
    for title, categories, chosen in (
        ("untruncated", LONGBENCH_E_CATEGORIES, untruncated_selections()),
        ("truncated to 31,500 tokens", LONGBENCH_E_TR_CATEGORIES, selections()),
    ):
        lines.append(f"-- {title}")
        lines += _table(simulations, categories, chosen)
    return "\n".join(lines)


def _table(simulations: list[dict], categories: dict, chosen: list[dict]) -> list[str]:
    longbench = {t for ts in categories.values() for t in ts}
    by_kv: dict[str, dict[str, float]] = {}
    for row in simulations:
        identity = row["identity"]
        if not set(identity["tasks"]) <= longbench or identity.get("limit") is not None:
            continue
        if "Llama-3.1-8B" not in identity["pretrained"]:
            continue
        key = json.dumps(identity.get("kv") or {"pipeline": []}, sort_keys=True)
        for task, metrics in row["scores"].items():
            if "score,none" in metrics:
                by_kv.setdefault(key, {})[task] = 100 * metrics["score,none"]
    names = {json.dumps({"pipeline": []}, sort_keys=True): "ours dense"}
    for selection in chosen:
        names[json.dumps(selection["kv"], sort_keys=True)] = "ours " + selection["tag"].removeprefix("turboquant_mse_")
    lines = []
    for key, label in names.items():
        scores = by_kv.get(key, {})
        missing = longbench - set(scores)
        if missing:
            lines.append(f"{label:40s}  missing {len(missing)} of {len(longbench)} tasks")
            continue
        means = [sum(scores[t] for t in ts) / len(ts) for ts in categories.values()]
        means.append(sum(scores.values()) / len(scores))
        lines.append(f"{label:40s}" + "".join(f"{v:10.2f}" for v in means))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="timing run: first N examples per task")
    parser.add_argument("--summary", action="store_true", help="print the table from results.json and exit")
    args = parser.parse_args()
    if args.summary:
        print(summary(ROOT / "results.json"))
        return

    from engine.eval_runner.progress import mirror_terminal

    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table

    mirror_terminal(ROOT)
    ensure_table(TABLE)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(longbench_run(args.limit), handle)
        path = Path(handle.name)
    command = [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"]
    if args.limit is not None:
        (ROOT / TIMING_ROOT).mkdir(parents=True, exist_ok=True)
        command += ["--results-root", str(ROOT / TIMING_ROOT)]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
