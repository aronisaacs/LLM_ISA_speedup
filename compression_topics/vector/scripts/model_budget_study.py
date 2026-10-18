#!/usr/bin/env python3
"""Per-scalar bitmask vector compression on Qwen3.5-9B and CodeLlama 7B at net budgets.

Same method as net_budget_study.py's per-scalar arm, calibrated per model:
WikiText singletons at 25/50/75% for every K and V slot of the layers that hold
a KV cache, then the greedy at 10/20/30/40/50% net of the bitmask, then tasks.

  qwen       Qwen3.5-9B: CEval, GSM8K, HumanEval-instruct. Hybrid model: only
             layers 3, 7, ..., 31 have a KV cache (16 slots); heads hold 256 values.
  codellama  CodeLlama 7B: plain HumanEval (base model), 64 slots.

Budgets are the mean net saving over the model's KV slots, so they are shares of
the whole KV cache.

Stages, in order:
  sweep   dense plus WikiText singletons for each KV slot at 25/50/75%
  greedy  selections at each net budget, and the run file
  pilot   one uniform 50% setting on a few items of each task, plus two WikiText
          singletons on a few documents; writes nothing (needs no sweep)
  tasks   dense and every selection on the model's tasks
  table   accuracy by budget from results.json, for whatever has finished

A rerun skips a simulation results.json already lists.

  python compression_topics/vector/scripts/model_budget_study.py --model qwen --only pilot
  python compression_topics/vector/scripts/model_budget_study.py --model qwen
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress  # noqa: E402
from catalog.models import BATCH_SIZE, CODELLAMA_7B, QWEN35_9B  # noqa: E402
from catalog.tasks import (  # noqa: E402
    CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_BASE, HUMANEVAL_INSTRUCT, WIKITEXT_FULL,
)
from compression_topics.vector.bitmask import net_saving_for  # noqa: E402
from engine.layer_select.budgets import budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

METHOD = "vector_compress"
TEMPLATE = vector_compress(prune_pct=25)
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
N_LAYERS = 32
# name: model_args, pretrained, head_dim, KV layers, tasks
MODELS = {
    "qwen": (QWEN35_9B, "Qwen/Qwen3.5-9B", 256, tuple(range(3, 32, 4)),
             (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT)),
    "codellama": (CODELLAMA_7B, "codellama/CodeLlama-7b-hf", 128, tuple(range(32)), (HUMANEVAL_BASE,)),
}
METRICS = {
    "ceval-valid": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1,create_test",
    "humaneval": "pass@1,create_test",
}
# Items per task in the pilot. CEval's limit applies to each of its subjects.
PILOT_LIMITS = {"ceval": 2, "gsm8k": 16, "humaneval_instruct": 8, "humaneval_base": 8}


def figures(model: str) -> str:
    """Relative to the repo root, so run files work on the Mac and on the DGX."""
    return f"compression_topics/vector/figures/models/{model}"


def _base(model: str) -> dict:
    return {"model": "hf", "batch_size": BATCH_SIZE, "model_args": MODELS[model][0]}


def sweep_run(model: str) -> dict:
    _, _, _, layers, _ = MODELS[model]
    configurations = expand_singleton_configs(n_layers=N_LAYERS, method_kv=TEMPLATE, method_tag="vector",
                                              results_dir=f"{figures(model)}/sweep", name_prefix=model,
                                              extra=WIKITEXT_FULL)
    kept = [c for c in configurations if "layer_slot" not in c or c["layer_slot"]["layer"] in layers]
    return {**_base(model), "configurations": kept}


def selections(model: str) -> list[dict]:
    _, pretrained, head_dim, layers, _ = MODELS[model]
    dense_ppl, rows = load_sweep_scores(method=METHOD, pretrained=pretrained)
    rows = [row for row in rows if row.slot.layer in layers]
    payloads = selections_for_budgets(rows, N_LAYERS, TEMPLATE, dense_ppl, budgets=BUDGETS,
                                      saving=net_saving_for(METHOD, head_dim), layers=layers)
    for payload in payloads:
        payload["tag"] = f"net_scalar_{payload['tag']}"
        payload["basis"] = "net of bitmask, mean over KV slots"
        payload["kv_layers"] = list(layers)
    return payloads


def write_greedy(model: str) -> None:
    payloads = selections(model)
    out = ROOT / figures(model)
    out.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (out / f"selected_{payload['tag']}.json").write_text(json.dumps(payload, indent=2) + "\n")
    tasks = MODELS[model][4]
    run = budget_run(payloads, tasks=tasks, results_dir=figures(model))
    run.update(_base(model))
    for configuration in run["configurations"]:
        configuration["name"] = configuration["name"].replace("llama31_", f"{model}_", 1)
    # multi_run deals configurations to GPUs by index; grouping by task spreads the slow ones.
    order = {task["name_task"]: rank for rank, task in enumerate(tasks)}
    run["configurations"].sort(key=lambda c: order[_task(model, c)["name_task"]])
    (out / "budgets_run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"{'selection':<20} {'net':>7} {'zeroed':>7}   K slots 25/50/75   V slots 25/50/75")
    for payload in payloads:
        levels = Counter((row["target"], row["level"]) for row in payload["assignment"])
        k = "/".join(f"{levels[('k', level)]:2d}" for level in (25, 50, 75))
        v = "/".join(f"{levels[('v', level)]:2d}" for level in (25, 50, 75))
        print(f"{payload['tag']:<20} {payload['compression']:7.4f} {payload['zeroed']:7.4f}   {k:>16}   {v:>16}")
    print(f"wrote  {out / 'budgets_run.json'}  ({len(run['configurations'])} configurations)")


def pilot_run(model: str) -> dict:
    """Uniform 50% on every KV slot over a few task items, and two short WikiText singletons."""
    _, _, _, layers, tasks = MODELS[model]
    uniform = vector_compress(k_layers=list(layers), v_layers=list(layers), prune_pct=50)
    configurations = []
    for task in tasks:
        extra = {key: value for key, value in task.items() if key not in {"name_task", "file"}}
        configurations.append({"name": f"{model}_{task['name_task']}_uniform50_pilot", "kv": uniform,
                               "output_path": f"{figures(model)}/pilot/{task['file']}_uniform50.json",
                               **extra, "limit": PILOT_LIMITS[task["name_task"]]})
    sweep = sweep_run(model)["configurations"]
    for configuration in (sweep[0], sweep[1]):  # dense and the first K slot at 25%
        pilot = copy.deepcopy(configuration)
        pilot.update(name=pilot["name"] + "_pilot", limit=4,
                     output_path=pilot["output_path"].replace("/sweep/", "/pilot/"))
        configurations.append(pilot)
    return {**_base(model), "configurations": configurations}


def table(model: str) -> None:
    from engine.eval_runner.cache import simulation_identity
    from engine.eval_runner.index import find_result
    from engine.kv_compress import parse_kv_spec

    run = json.loads((ROOT / figures(model) / "budgets_run.json").read_text())
    base = {key: value for key, value in run.items() if key != "configurations"}
    tasks = MODELS[model][4]
    scores = {}
    for configuration in run["configurations"]:
        row = find_result(simulation_identity(base, configuration, parse_kv_spec(configuration["kv"])))
        task = _task(model, configuration)
        tag = configuration["name"].split(f"{task['name_task']}_", 1)[1]
        metric = METRICS[task["tasks"][0]]
        scores[(tag, task["name_task"])] = None if row is None else row["scores"][task["tasks"][0]][metric]
    print(f"{model}{'':<14}" + "".join(f"{task['name_task']:>20}" for task in tasks))
    for tag in ["dense"] + [f"net_scalar_p{round(b * 100):02d}" for b in BUDGETS]:
        cells = [scores.get((tag, task["name_task"])) for task in tasks]
        print(f"{tag:<20}" + "".join(f"{'—':>20}" if c is None else f"{100 * c:19.1f}%" for c in cells))


def _task(model: str, configuration: dict) -> dict:
    return next(t for t in MODELS[model][4] if t["tasks"] == configuration["tasks"])


def _multi_run(run: dict, *extra: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run([sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), *extra],
                       cwd=ROOT, check=True)
    finally:
        path.unlink(missing_ok=True)


STAGES = ("sweep", "greedy", "pilot", "tasks", "table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--from", dest="start", default="sweep", choices=STAGES)
    parser.add_argument("--through", default="table", choices=STAGES)
    parser.add_argument("--only", choices=STAGES, help="run one stage")
    args = parser.parse_args()
    if args.only:
        stages = (args.only,)
    else:
        start, end = STAGES.index(args.start), STAGES.index(args.through)
        if start > end:
            raise SystemExit(f"--from {args.start} is after --through {args.through}")
        stages = tuple(s for s in STAGES[start:end + 1] if s != "pilot" or args.start == "pilot")
    for stage in stages:
        if stage == "sweep":
            _multi_run(sweep_run(args.model), "--skip-existing")
        elif stage == "greedy":
            write_greedy(args.model)
        elif stage == "pilot":
            _multi_run(pilot_run(args.model), "--dry-run")
        elif stage == "tasks":
            _multi_run(json.loads((ROOT / figures(args.model) / "budgets_run.json").read_text()), "--skip-existing")
        else:
            table(args.model)


if __name__ == "__main__":
    main()
