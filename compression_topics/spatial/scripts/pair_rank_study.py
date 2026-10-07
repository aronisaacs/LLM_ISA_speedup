#!/usr/bin/env python3
"""Keys-only pair merging at a fixed share of pairs per layer: WikiText sweep, greedy budgets, C-Eval.

Two simulations, both on the keys of Llama 3.1 8B Instruct (values are never touched):

  merge     a layer merges its most similar 25, 50, 75 or 100 percent of token pairs into their mean
  residual  the same, and each merged pair also keeps the top 1/4 of its difference exactly

Pairs are ranked within each layer across all KV heads, by the error left after the residual
(|rest| / |mean|). A key slot (one layer's keys) climbs 25, 50, 75, 100. The WikiText sweep scores
each slot at each rung alone. Greedy then raises the layer with the best compression per unit of
perplexity until the budget is met, and C-Eval (5-shot, prefill only) runs on each assignment.

Budgets are a share of the key cache bytes. A merged pair costs 1/2 of two dense keys with no
residual and 0.656 with the residual (mean, a 1-bit-per-feature mask, 32 kept differences), so the
residual simulation tops out at 34.4% and the merge-only one at 50%. A uniform run (every layer at
the same rung) is the control. Runs use batch size 1 because padding would rank as similar pairs.

  python compression_topics/spatial/scripts/pair_rank_study.py --variant merge --through sweep
  python compression_topics/spatial/scripts/pair_rank_study.py --variant residual
  python compression_topics/spatial/scripts/pair_rank_study.py --variant merge --from tasks
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

from catalog.compressions import pair_rank, pair_rank_residual  # noqa: E402
from catalog.models import LLAMA31_8B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.rungs import rungs_for  # noqa: E402
from engine.eval_runner.index import simulations  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_rank"
STAGES = ("sweep", "greedy", "tasks", "summary")
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
TARGETS = ("k",)
BATCH_SIZE = 1
CEVAL_METRIC = "acc,none"
VARIANTS = {
    "merge": {"method": "pair_rank", "kv": pair_rank(pct=25), "budgets": (0.10, 0.20, 0.30, 0.40)},
    "residual": {"method": "pair_rank_residual", "kv": pair_rank_residual(pct=25), "budgets": (0.10, 0.20, 0.30)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="summary", choices=STAGES)
    args = parser.parse_args()
    start, end = STAGES.index(args.start), STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    for stage in STAGES[start : end + 1]:
        _run_stage(stage, args.variant)


def _run_stage(stage: str, variant: str) -> None:
    out = FIGURES / variant
    if stage == "sweep":
        _multi_run(sweep_run(variant, out))
    elif stage == "greedy":
        out.mkdir(parents=True, exist_ok=True)
        run = task_run(variant, out, _payloads(variant))
        (out / "budgets_run.json").write_text(json.dumps(run, indent=2) + "\n")
        print(f"wrote  {out / 'budgets_run.json'}")
    elif stage == "tasks":
        _multi_run(json.loads((out / "budgets_run.json").read_text()))
    else:
        text = summary(variant)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(text + "\n")
        print(text)


def sweep_run(variant: str, out: Path) -> dict:
    """Dense plus every key layer at 25, 50, 75, 100 percent of its pairs, on WikiText."""
    spec = VARIANTS[variant]
    configurations = expand_singleton_configs(
        n_layers=LLAMA31_LAYERS,
        method_kv=spec["kv"],
        method_tag=spec["method"],
        results_dir=str(out),
        name_prefix=f"llama31_{variant}",
        extra=WIKITEXT_FULL,
        targets=TARGETS,
    )
    return {"model": "hf", "batch_size": BATCH_SIZE, "model_args": LLAMA31_8B, "configurations": configurations}


def _payloads(variant: str) -> list[dict]:
    spec = VARIANTS[variant]
    dense_ppl, rows = load_sweep_scores(method=spec["method"], pretrained=PRETRAINED)
    payloads = selections_for_budgets(
        rows, LLAMA31_LAYERS, spec["kv"], dense_ppl, budgets=spec["budgets"], targets=TARGETS
    )
    for payload in payloads:
        payload["tag"] = f"{variant}_{payload['tag']}"
    return payloads


def task_run(variant: str, out: Path, payloads: list[dict]) -> dict:
    """Dense, greedy at each budget, and the uniform control (every key layer at one rung), on C-Eval."""
    run = budget_run(payloads, tasks=(CEVAL_VALID_5SHOT,), results_dir=str(out))
    run["batch_size"] = BATCH_SIZE
    spec = VARIANTS[variant]
    method = pair_rank if spec["method"] == "pair_rank" else pair_rank_residual
    extra = {key: value for key, value in CEVAL_VALID_5SHOT.items() if key not in {"name_task", "file"}}
    for rung in rungs_for(spec["method"]):
        tag = f"{variant}_uniform_p{rung.level}"
        run["configurations"].append(
            {
                "name": f"llama31_ceval_{tag}",
                "kv": method(pct=rung.level),
                "output_path": f"{out}/ceval_{tag}.json",
                "metadata": {"kv_budget": rung.fraction, "kv_compression": rung.fraction},
                **extra,
            }
        )
    return run


def summary(variant: str, root: Path | None = None) -> str:
    """C-Eval accuracy against the share of key bytes removed, greedy next to uniform. Read from results.json."""
    method = VARIANTS[variant]["method"]
    dense = None
    rows = []
    for record in simulations(root):
        identity = record.get("identity") or {}
        if identity.get("tasks") != ["ceval-valid"] or identity.get("pretrained") != PRETRAINED:
            continue
        score = (record.get("scores") or {}).get("ceval-valid", {}).get(CEVAL_METRIC)
        pipeline = (identity.get("kv") or {}).get("pipeline") or []
        if score is None:
            continue
        if not pipeline:
            dense = float(score)
            continue
        if any(step.get("method") != method for step in pipeline):
            continue
        uniform = len(pipeline) == 1 and pipeline[0].get("k_layers") == "all"
        rows.append((float(record.get("compression", 0.0)), "uniform" if uniform else "greedy", float(score), record.get("budget")))
    lines = [
        f"C-Eval ({CEVAL_METRIC}), Llama 3.1 8B Instruct, keys only, pair_rank '{variant}'"
        + ("" if dense is None else f"; dense {dense:.3f}"),
        "",
        "| allocation | budget | key bytes removed | accuracy | vs dense |",
        "|---|---|---|---|---|",
    ]
    if not rows:
        return "\n".join(lines[:1] + ["", "no finished C-Eval runs in results.json"])
    for compression, kind, score, budget in sorted(rows, key=lambda row: (row[1], row[0])):
        change = "" if dense is None else f"{score - dense:+.3f}"
        shown = "" if budget is None or kind == "uniform" else f"{budget:.2f}"
        lines.append(f"| {kind} | {shown} | {compression:.3f} | {score:.3f} | {change} |")
    return "\n".join(lines)


def _multi_run(run: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"],
            cwd=ROOT,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
