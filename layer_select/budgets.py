"""Greedy assignments at fixed mean-compression budgets, and the task run they become."""

from __future__ import annotations

import json
from pathlib import Path

from catalog.compressions import DENSE
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT
from layer_select.apply import kv_for_assignment, method_template
from layer_select.greedy.rank_fill import rank_fill
from layer_select.levels import LEVELS, mean_compression
from layer_select.scores import kv_from_payload, load_sweep_scores
from layer_select.slots import all_slots

LLAMA31_LAYERS = 32
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
TASKS = (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT)

SWEEP_RESULTS = "results/vector_sweep"
STUDY_DIR = "results/vector_study"
BUDGET_RESULTS = "results/vector_budgets"


def selections_for_budgets(rows, n_layers, method_kv, dense_ppl, budgets=BUDGETS):
    """One rank-fill payload per budget. ``compression`` is the realized mean."""
    template = method_template(method_kv)
    n_slots = len(all_slots(n_layers))
    payloads = []
    for budget in budgets:
        assignment = rank_fill(rows, n_layers, budget, dense_ppl=dense_ppl)
        payloads.append(
            {
                "tag": _budget_tag(budget),
                "budget": budget,
                "compression": mean_compression(assignment, n_slots),
                "n_layers": n_layers,
                "levels": list(LEVELS),
                "dense_ppl": dense_ppl,
                "assignment": [
                    {**slot.to_dict(), "level": pct} for slot, pct in sorted(assignment.items())
                ],
                "kv": kv_for_assignment(template, assignment),
            }
        )
    return payloads


def write_selections(scores_dir, out_dir, n_layers=None, budgets=BUDGETS) -> list[dict]:
    """Read a sweep directory and write ``selected_pXX.json`` plus ``budgets.json``."""
    dense_ppl, rows = load_sweep_scores(scores_dir)
    if n_layers is None:
        n_layers = max(row.slot.layer for row in rows) + 1
    method_kv = kv_from_payload(json.loads(Path(rows[0].path).read_text()))
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


def budget_run(selections, tasks=TASKS) -> dict:
    """Dense plus one configuration per selection, for each task."""
    configurations = []
    for task in tasks:
        configurations.append(_task_configuration(task, "dense", DENSE, budget=0.0, compression=0.0))
    for selection in selections:
        for task in tasks:
            configurations.append(
                _task_configuration(
                    task,
                    selection["tag"],
                    selection["kv"],
                    budget=selection["budget"],
                    compression=selection["compression"],
                )
            )
    return {
        "model": "hf",
        "batch_size": "auto:4",
        "model_args": LLAMA31_8B,
        "configurations": configurations,
    }


def write_budget_run(selections, path, tasks=TASKS) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(budget_run(selections, tasks=tasks), indent=2) + "\n")
    return destination


def _task_configuration(task, tag, kv, budget, compression) -> dict:
    extra = {key: value for key, value in task.items() if key not in {"name_task", "file"}}
    configuration = {
        "name": f"llama31_{task['name_task']}_{tag}",
        "kv": kv,
        "output_path": f"{BUDGET_RESULTS}/{task['file']}_{tag}.json",
        "metadata": {"kv_budget": budget, "kv_compression": compression},
    }
    configuration.update(extra)
    return configuration


def _budget_tag(budget: float) -> str:
    return f"p{round(budget * 100):02d}"
