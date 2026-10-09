#!/usr/bin/env python3
"""Stage B for ``group_rd``: C-Eval and WikiText with per-block formats on keys and values.

Every layer's keys and values get the same saving (30, 40, 50%), a format per
4-token block from one 8-format menu, chosen with one price per slot. Two arms
differ only in which residual entries keys keep:
  plain       the largest deviations (as every method so far)
  attention   the largest deviations weighted by query energy (``select_by='query'``),
              i.e. the entries that move attention scores most
Keys price formats by query-weighted error, values by squared error; the query
weights come from the stage-A calibration capture (WikiText calibration chunks,
not the evaluated tasks). Dense rows are included and reused from results.json
when present. Results also go to the repository results.json.

  python compression_topics/spatial/scripts/group_rd_tasks_study.py              # plan only
  python compression_topics/spatial/scripts/group_rd_tasks_study.py --execute
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts.group_rd_offline import CANDIDATE_MENUS
from engine.eval_runner.files import write_json
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration

OUT = ROOT / "compression_topics/spatial/figures/group_rd_tasks"
QUERY_WEIGHTS = "compression_topics/spatial/figures/group_rd_offline/query_weights.pt"
BUDGETS = (.3, .4, .5)
ARMS = ("plain", "attention")
TASKS = {"ceval": CEVAL_VALID_5SHOT, "wikitext": WIKITEXT_FULL}
PREVIOUS = ROOT / "compression_topics/spatial/figures/pair_quad_shared_sampled/summary.json"


def kv(budget, arm, menu, granularity=4, query_weights=QUERY_WEIGHTS):
    common = {"method": "group_rd", "accounting": group_rd.ACCOUNTING, "saving": budget,
              "menu": list(menu), "granularity": granularity}
    keys = {**common, "k_layers": "all", "v_layers": [], "distortion": "query",
            "query_weights": query_weights, "select_by": "query" if arm == "attention" else "deviation"}
    values = {**common, "k_layers": [], "v_layers": "all", "distortion": "squared"}
    return {"pipeline": [keys, values]}


def plan(out, *, budgets=BUDGETS, menu_name="flag_d", granularity=4, model_args=LLAMA31_8B, tasks=tuple(TASKS)):
    menu = CANDIDATE_MENUS[menu_name]
    group_rd.mode_names(menu=menu)  # validate before anything runs
    configs = [configuration(f"dense_{task}", {"pipeline": []}, out / "tasks", TASKS[task]) for task in tasks]
    rows = []
    for arm in ARMS:
        for budget in budgets:
            spec = kv(budget, arm, menu, granularity)
            parse_kv_spec(spec)
            for task in tasks:
                tag = f"{arm}_b{round(budget * 100):02d}_{task}"
                configs.append(configuration(tag, spec, out / "tasks", TASKS[task], kv_budget=budget, arm=arm,
                                             menu=menu_name, granularity=granularity))
                rows.append({"tag": tag, "arm": arm, "budget": budget, "task": task})
    return {"menu_name": menu_name, "menu": menu, "granularity": granularity, "budgets": list(budgets),
            "arms": list(ARMS), "tasks": list(tasks), "query_weights": QUERY_WEIGHTS, "rows": rows,
            "run": {"model": "hf", "model_args": model_args, "batch_size": 1, "configurations": configs}}


def execute(path):
    subprocess.run([sys.executable, str(ROOT / "engine/multi_run.py"), "--run", str(path),
                    "--skip-existing", "--write-results", "--results-root", str(ROOT)], cwd=ROOT, check=True)


def score(payload, task):
    results = payload["results"]
    if task == "ceval":
        return results["ceval-valid"]["acc,none"]
    return results["wikitext"]["word_perplexity,none"]


def summary(out, manifest):
    tasks = out / "tasks"
    dense = {task: score(json.loads((tasks / f"dense_{task}.json").read_text()), task) for task in manifest["tasks"]}
    rows = []
    for row in manifest["rows"]:
        path = tasks / f"{row['tag']}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        storage = payload.get("storage", {}).get("targets", {})
        value = score(payload, row["task"])
        rows.append({**row, "score": value, "delta": value - dense[row["task"]],
                     "measured_kv_saving": storage.get("kv", {}).get("compression"),
                     "measured_k_saving": storage.get("k", {}).get("compression"),
                     "measured_v_saving": storage.get("v", {}).get("compression")})
    previous = json.loads(PREVIOUS.read_text()) if PREVIOUS.exists() else {"rows": []}
    pair_quad = {round(r["budget"] * 100): r for r in previous["rows"] if r["mode"] == "kv"}
    lines = ["# group_rd stage B: C-Eval and WikiText", "",
             f"Menu `{manifest['menu_name']}`: {', '.join(manifest['menu'])}; a format per "
             f"{manifest['granularity']} tokens; same saving in every key and value layer.", ""]
    if "ceval" in dense:
        lines += [f"C-Eval valid, 5-shot; dense {dense['ceval']:.2%}. Change in points; measured whole-KV saving in brackets.", "",
                  "| saving | plain residuals | attention-aware residuals | pair/quad per layer (previous) | pairs only (previous) |",
                  "|---|---|---|---|---|"]
        for budget in manifest["budgets"]:
            cells = []
            for arm in ARMS:
                hit = [r for r in rows if r["arm"] == arm and r["budget"] == budget and r["task"] == "ceval"]
                cells.append(f"{hit[0]['delta'] * 100:+.2f} ({hit[0]['measured_kv_saving']:.1%})" if hit else "–")
            b = round(budget * 100)
            for policy in ("mixed", "pairs"):
                match = [r for r in previous["rows"] if r["mode"] == "kv" and r["policy"] == policy and round(r["budget"] * 100) == b]
                cells.append(f"{match[0]['delta_accuracy'] * 100:+.2f}" if match else "–")
            lines.append(f"| {budget:.0%} | " + " | ".join(cells) + " |")
        lines.append("")
    if "wikitext" in dense:
        lines += [f"WikiText word perplexity; dense {dense['wikitext']:.3f}.", "",
                  "| saving | plain residuals | attention-aware residuals |", "|---|---|---|"]
        for budget in manifest["budgets"]:
            cells = []
            for arm in ARMS:
                hit = [r for r in rows if r["arm"] == arm and r["budget"] == budget and r["task"] == "wikitext"]
                cells.append(f"{hit[0]['score']:.3f} ({hit[0]['delta']:+.3f})" if hit else "–")
            lines.append(f"| {budget:.0%} | " + " | ".join(cells) + " |")
    report = {"dense": dense, "rows": rows, "menu": manifest["menu"], "menu_name": manifest["menu_name"]}
    write_json(out / "summary.json", report)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    parser.add_argument("--menu", choices=sorted(CANDIDATE_MENUS), default="flag_d")
    parser.add_argument("--granularity", type=int, choices=(4, 16, 64), default=4)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--model-args", default=LLAMA31_8B)
    parser.add_argument("--summary-only", action="store_true", help="rebuild summary.md from finished runs")
    parser.add_argument("--execute", action="store_true", help="start the evaluations; otherwise only write the plan")
    args = parser.parse_args()
    if any(not 0 < b < .74 for b in args.budgets):
        parser.error("budgets must be between 0 and 0.74")
    out = args.out.resolve()
    if not (ROOT / QUERY_WEIGHTS).exists():
        parser.error(f"missing {QUERY_WEIGHTS}; run the stage-A capture first")
    manifest = plan(out, budgets=tuple(sorted(set(args.budgets))), menu_name=args.menu, granularity=args.granularity,
                    model_args=args.model_args, tasks=tuple(args.tasks))
    if args.summary_only:
        summary(out, manifest)
        return
    write_json(out / "manifest.json", manifest)
    write_json(out / "tasks_run.json", manifest["run"])
    runs = len(manifest["run"]["configurations"])
    print(f"Plan: {runs} runs ({len(manifest['tasks'])} dense + {len(manifest['rows'])} group_rd) on "
          f"{', '.join(manifest['tasks'])}; arms {', '.join(ARMS)}; savings "
          f"{', '.join(f'{b:.0%}' for b in manifest['budgets'])} in every key and value layer.")
    print(f"Menu {args.menu}: {', '.join(manifest['menu'])}; a format per {args.granularity} tokens.")
    print(f"Output: {out}; scores also go to results.json")
    if not args.execute:
        print("Plan only. Add --execute when ready to run.")
        return
    execute(out / "tasks_run.json")
    summary(out, manifest)


if __name__ == "__main__":
    main()
