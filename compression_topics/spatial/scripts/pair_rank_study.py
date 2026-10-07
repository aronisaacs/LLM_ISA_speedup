#!/usr/bin/env python3
"""Keys-only WikiText sweep, greedy budgets, and C-Eval for merging the most similar token pairs.

Two variants, each with its own sweep and greedy fill:
  merge      pair_rank: merge the given share of a layer's key pairs, most similar first, no residual
  residual   pair_rank_residual: the same, keeping 1/4 of each merged pair's largest differences exactly

Only key slots are swept and filled. Each of the 32 key layers is run alone at 25, 50, 75 and 100%
of its pairs merged. The greedy fill then climbs those rungs until the mean share of key-cache bytes
removed reaches each budget; a residual's bytes count against the budget (a merged pair keeps
0.65625 of its bytes instead of 0.5). Budgets are shares of the key cache, not the whole KV cache.
The residual variant removes at most 34.4% of the key cache, so its 40% budget is skipped.

Pairs are ranked within each sequence, so every run uses batch size 1: padding from a batch would
take part in the ranking. Keys are compared as stored, with RoPE applied. C-Eval is scored by
loglikelihood, so it is prefill only. Scores land in results.json; a finished run is skipped.

  python compression_topics/spatial/scripts/pair_rank_study.py                        # both variants, every stage
  python compression_topics/spatial/scripts/pair_rank_study.py --variant merge --through sweep
  python compression_topics/spatial/scripts/pair_rank_study.py --from summary
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
from catalog.models import LLAMA31_8B, LLAMA32_1B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL  # noqa: E402
from engine.layer_select.budgets import budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.rungs import rungs_for  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402
from engine.layer_select.slots import pretrained_from_model_args  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_rank"
STAGES = ("sweep", "greedy", "tasks", "summary")
BUDGETS = (0.10, 0.20, 0.30, 0.40)  # share of key-cache bytes removed
TARGETS = ("k",)
VARIANTS = {
    "merge": ("pair_rank", pair_rank(k_layers=[], v_layers=[])),
    "residual": ("pair_rank_residual", pair_rank_residual(k_layers=[], v_layers=[])),
}
MODELS = {"8b": (LLAMA31_8B, 32, "llama31"), "1b": (LLAMA32_1B, 16, "llama32_1b")}
CEVAL_METRIC = "acc,none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="summary", choices=STAGES)
    parser.add_argument("--model", choices=sorted(MODELS), default="8b", help="1b is for a quick local check")
    parser.add_argument("--limit", type=int, default=None, help="cap documents/questions per run (local checks only)")
    args = parser.parse_args()
    start, end = STAGES.index(args.start), STAGES.index(args.through)
    if start > end:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    variants = tuple(VARIANTS) if args.variant == "both" else (args.variant,)
    for stage in STAGES[start : end + 1]:
        for variant in variants:
            _run_stage(stage, variant, args)


def _run_stage(stage: str, variant: str, args) -> None:
    model_args, n_layers, prefix = MODELS[args.model]
    folder = FIGURES / args.model / variant
    method, method_kv = VARIANTS[variant]
    if stage == "sweep":
        extra = dict(WIKITEXT_FULL)
        if args.limit is not None:
            extra["limit"] = args.limit
        configurations = expand_singleton_configs(
            n_layers=n_layers,
            method_kv=method_kv,
            method_tag=method,
            results_dir=str(folder / "sweep"),
            name_prefix=prefix,
            extra=extra,
            targets=TARGETS,
        )
        _multi_run({"model": "hf", "batch_size": 1, "model_args": model_args, "configurations": configurations})
        return
    if stage == "greedy":
        payloads = _payloads(variant, args)
        run = budget_run(payloads, tasks=(_ceval(args),), results_dir=str(folder / "tasks"))
        run["batch_size"] = 1
        run["model_args"] = model_args
        for configuration in run["configurations"]:
            configuration["name"] = configuration["name"].replace("llama31", prefix, 1)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "budgets_run.json").write_text(json.dumps(run, indent=2) + "\n")
        (folder / "selections.json").write_text(json.dumps(payloads, indent=2) + "\n")
        print(f"wrote  {folder / 'budgets_run.json'}")
        return
    if stage == "tasks":
        _multi_run(json.loads((folder / "budgets_run.json").read_text()))
        return
    text = summary(variant, args)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.md").write_text(text + "\n")
    print(text)


def _ceval(args) -> dict:
    task = dict(CEVAL_VALID_5SHOT)
    if args.limit is not None:
        task["limit"] = args.limit
    return task


def _payloads(variant: str, args) -> list[dict]:
    model_args, n_layers, _ = MODELS[args.model]
    method, method_kv = VARIANTS[variant]
    dense_ppl, rows = load_sweep_scores(method=method, pretrained=pretrained_from_model_args(model_args))
    rows = [row for row in rows if row.slot.target in TARGETS]
    ceiling = max(rung.fraction for rung in rungs_for(method))
    budgets = tuple(budget for budget in BUDGETS if budget <= ceiling + 1e-9)
    for budget in BUDGETS:
        if budget not in budgets:
            print(f"skip   {variant} budget {budget:.0%}: this variant removes at most {ceiling:.1%} of the key cache")
    payloads = selections_for_budgets(rows, n_layers, method_kv, dense_ppl, budgets=budgets, targets=TARGETS)
    for payload in payloads:
        payload["tag"] = f"{method}_keys_{payload['tag']}"
    return payloads


def summary(variant: str, args) -> str:
    """Sweep ΔPPL per key layer and rung, then C-Eval accuracy per budget."""
    model_args, n_layers, _ = MODELS[args.model]
    method, _ = VARIANTS[variant]
    lines = [f"## {variant}: {method}, keys only, {args.model}", ""]
    try:
        dense_ppl, rows = load_sweep_scores(method=method, pretrained=pretrained_from_model_args(model_args))
    except ValueError as error:
        return "\n".join(lines + [f"no sweep scores yet ({error})"])
    table = {(row.slot.layer, row.level): row.ppl for row in rows if row.slot.target in TARGETS}
    levels = [rung.level for rung in rungs_for(method)]
    lines += [f"WikiText word perplexity, dense {dense_ppl:.4f}. Change when one key layer merges that share of its pairs:", ""]
    lines.append("| layer | " + " | ".join(f"{level}%" for level in levels) + " |")
    lines.append("|---|" + "---|" * len(levels))
    for layer in range(n_layers):
        cells = [f"{table[(layer, level)] - dense_ppl:+.4f}" if (layer, level) in table else "-" for level in levels]
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    folder = FIGURES / args.model / variant / "tasks"
    results = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        payload = json.loads(path.read_text())
        score = next((m.get(CEVAL_METRIC) for m in (payload.get("results") or {}).values() if isinstance(m, dict) and m.get(CEVAL_METRIC) is not None), None)
        if score is None:
            continue
        metadata = next((c.get("metadata") or {} for c in (payload.get("configs") or {}).values() if isinstance(c, dict)), {})
        results.append((float(metadata.get("kv_budget", 0.0)), float(metadata.get("kv_compression", 0.0)), float(score), payload.get("rank_stats")))
    lines += ["", f"C-Eval valid, 5-shot ({CEVAL_METRIC}). Budget and achieved share are of key-cache bytes:", ""]
    if not results:
        return "\n".join(lines + ["no C-Eval runs yet"])
    lines += ["| budget | key bytes removed | pairs merged (all key layers) | accuracy |", "|---|---|---|---|"]
    for budget, compression, score, stats in sorted(results):
        merged = "-"
        if stats and stats.get("k", {}).get("pairs"):
            merged = f"{stats['k']['merged'] / stats['k']['pairs']:.3f}"
        lines.append(f"| {budget:.0%} | {compression:.3f} | {merged} | {score:.4f} |")
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
