"""Greedy assignments at fixed mean-compression budgets, and the task run they become."""

from __future__ import annotations

import json
from pathlib import Path

from catalog.compressions import DENSE
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT
from engine.layer_select.apply import kv_for_assignment, method_template
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import mean_compression
from engine.layer_select.rungs import rungs_for
from engine.layer_select.scores import kv_from_payload, load_sweep_scores
from engine.layer_select.slots import all_slots

LLAMA31_LAYERS = 32
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
TASKS = (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT)
FIGURES = "figures"


def selections_for_budgets(rows, n_layers, method_kv, dense_ppl, budgets=BUDGETS):
    """One rank-fill payload per budget. ``compression`` is the realized mean."""
    template = method_template(method_kv)
    rungs = rungs_for(template["pipeline"][0]["method"])
    n_slots = len(all_slots(n_layers))
    payloads = []
    for budget in budgets:
        assignment = rank_fill(rows, n_layers, budget, dense_ppl=dense_ppl, rungs=rungs)
        payloads.append(
            {
                "tag": _budget_tag(budget),
                "budget": budget,
                "compression": mean_compression(assignment, n_slots, rungs),
                "n_layers": n_layers,
                "levels": [rung.level for rung in rungs],
                "dense_ppl": dense_ppl,
                "assignment": [
                    {**slot.to_dict(), "level": pct} for slot, pct in sorted(assignment.items())
                ],
                "kv": kv_for_assignment(template, assignment),
            }
        )
    return payloads


def write_selections(scores_dir, out_dir, n_layers=None, budgets=BUDGETS, scored=None) -> list[dict]:
    """Read a sweep directory, or ``scored`` from the index, and write selections."""
    dense_ppl, rows = scored if scored is not None else load_sweep_scores(scores_dir)
    if n_layers is None:
        n_layers = max(row.slot.layer for row in rows) + 1
    method_kv = rows[0].kv if rows[0].kv is not None else kv_from_payload(json.loads(Path(rows[0].path).read_text()))
    payloads = selections_for_budgets(rows, n_layers, method_kv, dense_ppl, budgets=budgets)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        (out / f"selected_{payload['tag']}.json").write_text(json.dumps(payload, indent=2) + "\n")
    index = [
        {"tag": "dense", "budget": 0.0, "compression": 0.0},
        *[
            {"tag": payload["tag"], "budget": payload["budget"], "compression": payload["compression"]}
            for payload in payloads
        ],
    ]
    (out / "budgets.json").write_text(json.dumps(index, indent=2) + "\n")
    return payloads


def budget_run(selections, tasks=TASKS, results_dir=FIGURES) -> dict:
    """Dense plus one configuration per selection, for each task."""
    configurations = []
    for task in tasks:
        configurations.append(
            _task_configuration(task, "dense", DENSE, budget=0.0, compression=0.0, results_dir=results_dir)
        )
    for selection in selections:
        for task in tasks:
            configurations.append(
                _task_configuration(
                    task,
                    selection["tag"],
                    selection["kv"],
                    budget=selection["budget"],
                    compression=selection["compression"],
                    results_dir=results_dir,
                )
            )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


def write_budget_run(selections, path, tasks=TASKS, results_dir=FIGURES) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(budget_run(selections, tasks=tasks, results_dir=results_dir), indent=2) + "\n")
    return destination


def _task_configuration(task, tag, kv, budget, compression, results_dir=FIGURES) -> dict:
    extra = {key: value for key, value in task.items() if key not in {"name_task", "file"}}
    configuration = {
        "name": f"llama31_{task['name_task']}_{tag}",
        "kv": kv,
        "output_path": f"{results_dir.rstrip('/')}/{task['file']}_{tag}.json",
        "metadata": {"kv_budget": budget, "kv_compression": compression},
    }
    configuration.update(extra)
    return configuration


def _budget_tag(budget: float) -> str:
    return f"p{round(budget * 100):02d}"
