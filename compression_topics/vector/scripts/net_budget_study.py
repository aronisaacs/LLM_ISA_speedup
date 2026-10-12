#!/usr/bin/env python3
"""Vector compression of keys and values at net budgets after the bitmask.

The greedy is the usual rank-fill over the WikiText singleton sweeps, but each
rung counts its saving net of the bitmask (compression_topics/vector/bitmask.py):
a 40% budget means 40% of the KV cache bits saved once the mask is stored.

Two sets of arms, each with its own run file:
  plain   per-scalar and RoPE-paired (budgets_run.json)
  query   the same with query-aware key ranking (budgets_run_query.json)

The query arms only change keys, so they get their own WikiText sweep of the
32 key slots; their value slots compute exactly what the plain arms' value
slots do, and reuse the plain sweep scores.

Stages, in order:
  sweep   WikiText singletons at 25/50/75% for each key slot (query arms only)
  greedy  selections at 20/30/35/40/45/50/55% net for each arm, and the run file
  pilot   the 50% selections on a few items of each task; writes nothing
  tasks   CEval, GSM8K (all 1319) and HumanEval for dense and every selection
  table   accuracy by budget from results.json, for whatever has finished

A rerun skips a simulation results.json already lists.

  python compression_topics/vector/scripts/net_budget_study.py --through greedy
  python compression_topics/vector/scripts/net_budget_study.py --only pilot
  python compression_topics/vector/scripts/net_budget_study.py --from tasks
  python compression_topics/vector/scripts/net_budget_study.py --arms query --from sweep
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

from catalog.compressions import vector_compress, vector_compress_pair, vector_compress_query  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT, WIKITEXT_FULL  # noqa: E402
from compression_topics.vector.bitmask import net_saving_for  # noqa: E402
from engine.layer_select.budgets import LLAMA31_LAYERS, budget_run, selections_for_budgets  # noqa: E402
from engine.layer_select.scores import load_sweep_scores  # noqa: E402
from engine.layer_select.sweep import expand_singleton_configs  # noqa: E402

# Relative to the repo root, so the run file works on the Mac and on the DGX.
FIGURES_REL = "compression_topics/vector/figures/net_bitmask"
FIGURES = ROOT / FIGURES_REL
STAGES = ("sweep", "greedy", "pilot", "tasks", "table")
PRETRAINED = "meta-llama/Llama-3.1-8B-Instruct"
BUDGETS = (0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
TASKS = (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT)
# name: (method, template, method whose sweep scores the value slots)
ARMS = {
    "plain": {
        "scalar": ("vector_compress", vector_compress(prune_pct=25), "vector_compress"),
        "pair": ("vector_compress_pair", vector_compress_pair(prune_pct=25), "vector_compress_pair"),
    },
    "query": {
        "scalar_query": ("vector_compress_query", vector_compress_query(prune_pct=25), "vector_compress"),
        "pair_query": ("vector_compress_pair_query", vector_compress_query(prune_pct=25, pairs=True),
                       "vector_compress_pair"),
    },
}
RUNS = {"plain": FIGURES / "budgets_run.json", "query": FIGURES / "budgets_run_query.json"}
# Items per task in the pilot. CEval's limit applies to each of its subjects.
PILOT_LIMITS = {"ceval": 2, "gsm8k": 32, "humaneval_instruct": 16}
PILOT_TAG = "p50"
METRICS = {
    "ceval-valid": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1,create_test",
}


def sweep_run(arms: str) -> dict:
    """Dense, then each key slot at 25/50/75% for every arm that needs its own key sweep."""
    configurations = []
    for name, (method, template, value_source) in ARMS[arms].items():
        if value_source == method:
            continue
        configurations += expand_singleton_configs(n_layers=LLAMA31_LAYERS, method_kv=template, method_tag=name,
                                                   results_dir=f"{FIGURES_REL}/sweep", name_prefix="llama31",
                                                   extra=WIKITEXT_FULL, targets=("k",))
    unique = {configuration["name"]: configuration for configuration in configurations}
    return {"model": "hf", "batch_size": BATCH_SIZE, "model_args": LLAMA31_8B,
            "configurations": list(unique.values())}


def sweep_rows(method: str, value_source: str):
    dense_ppl, rows = load_sweep_scores(method=method, pretrained=PRETRAINED)
    if value_source != method:
        _, values = load_sweep_scores(method=value_source, pretrained=PRETRAINED)
        rows = [row for row in rows if row.slot.target == "k"] + [row for row in values if row.slot.target == "v"]
    return dense_ppl, rows


def selections(arms: str) -> list[dict]:
    payloads = []
    for name, (method, template, value_source) in ARMS[arms].items():
        dense_ppl, rows = sweep_rows(method, value_source)
        for payload in selections_for_budgets(rows, LLAMA31_LAYERS, template, dense_ppl, budgets=BUDGETS,
                                              saving=net_saving_for(method)):
            payload["tag"] = f"net_{name}_{payload['tag']}"
            payload["basis"] = "net of bitmask"
            payloads.append(payload)
    return payloads


def write_greedy(arms: str) -> list[dict]:
    payloads = selections(arms)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (FIGURES / f"selected_{payload['tag']}.json").write_text(json.dumps(payload, indent=2) + "\n")
    run = budget_run(payloads, tasks=TASKS, results_dir=FIGURES_REL)
    # multi_run deals configurations to GPUs by index, so group them by task: each
    # GPU then gets a share of the slow GSM8K runs instead of one GPU getting all.
    order = {task["name_task"]: rank for rank, task in enumerate(TASKS)}
    run["configurations"].sort(key=lambda c: order[next(t["name_task"] for t in TASKS if t["tasks"] == c["tasks"])])
    RUNS[arms].write_text(json.dumps(run, indent=2) + "\n")
    print(f"{'selection':<22} {'net':>7} {'zeroed':>7}   K slots 25/50/75   V slots 25/50/75")
    for payload in payloads:
        levels = Counter((row["target"], row["level"]) for row in payload["assignment"])
        k = "/".join(f"{levels[('k', level)]:2d}" for level in (25, 50, 75))
        v = "/".join(f"{levels[('v', level)]:2d}" for level in (25, 50, 75))
        print(f"{payload['tag']:<22} {payload['compression']:7.4f} {payload['zeroed']:7.4f}   {k:>16}   {v:>16}")
    print(f"wrote  {RUNS[arms]}  ({len(run['configurations'])} configurations)")
    return payloads


def pilot_run(arms: str) -> dict:
    run = json.loads(RUNS[arms].read_text())
    configurations = []
    for configuration in run["configurations"]:
        if not configuration["name"].endswith(f"_{PILOT_TAG}"):
            continue
        task = next(t for t in TASKS if configuration["tasks"] == t["tasks"])
        pilot = copy.deepcopy(configuration)
        pilot["limit"] = PILOT_LIMITS[task["name_task"]]
        pilot["name"] += "_pilot"
        pilot["output_path"] = pilot["output_path"].replace(".json", "_pilot.json")
        configurations.append(pilot)
    expected = len(ARMS[arms]) * len(TASKS)
    if len(configurations) != expected:
        raise SystemExit(f"expected {expected} pilot configurations, found {len(configurations)}")
    return {**run, "configurations": configurations}


def table(arms: str) -> None:
    from engine.eval_runner.cache import simulation_identity
    from engine.eval_runner.index import find_result
    from engine.kv_compress import parse_kv_spec

    run = json.loads(RUNS[arms].read_text())
    base = {key: value for key, value in run.items() if key != "configurations"}
    scores: dict[tuple[str, str], float | None] = {}
    for configuration in run["configurations"]:
        identity = simulation_identity(base, configuration, parse_kv_spec(configuration["kv"]))
        row = find_result(identity)
        task = configuration["tasks"][0]
        value = None if row is None else row["scores"][task][METRICS[task]]
        scores[(configuration["name"].split(f"{_short(task)}_", 1)[1], task)] = value
    print(f"{'':<22}" + "".join(f"{_short(task):>20}" for task in METRICS))
    for tag in ["dense"] + [f"net_{name}_p{round(b * 100):02d}" for name in ARMS[arms] for b in BUDGETS]:
        cells = [scores.get((tag, task)) for task in METRICS]
        print(f"{tag:<22}" + "".join(f"{'—':>20}" if cell is None else f"{100 * cell:19.1f}%" for cell in cells))


def _short(task: str) -> str:
    return next(t["name_task"] for t in TASKS if t["tasks"] == [task])


def _multi_run(run: dict, *extra: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run([sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), *extra],
                       cwd=ROOT, check=True)
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="plain", choices=tuple(ARMS))
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
        # The pilot is its own step; a full run goes straight to tasks.
        stages = tuple(s for s in STAGES[start:end + 1] if s != "pilot" or args.start == "pilot")
    for stage in stages:
        if stage == "sweep":
            run = sweep_run(args.arms)
            if run["configurations"]:
                _multi_run(run, "--skip-existing")
        elif stage == "greedy":
            write_greedy(args.arms)
        elif stage == "pilot":
            _multi_run(pilot_run(args.arms), "--dry-run")
        elif stage == "tasks":
            _multi_run(json.loads(RUNS[args.arms].read_text()), "--skip-existing")
        else:
            table(args.arms)


if __name__ == "__main__":
    main()
