#!/usr/bin/env python3
"""Spatial ladder at 30/40/50%.

Arms (each calibrated and allocated independently):
* pairs: dense or pairs with 0/16/32 residual entries;
* hierarchy: the same pairs plus quads and eights with 0/16 entries;
* query_pairs: pairs with query-aware loss and residuals on keys;
* query_hierarchy: the hierarchy with query-aware loss and residuals on keys.

Simplifications of query_pairs (each calibrated on its own):
* query_pairs_per_head: every head meets the saving alone (no shared window budget);
* query_pairs_no_rope: keys are merged without undoing RoPE;
* query_pairs_no_norms: no per-token norms; raw vectors are merged;
* query_pairs_no_rope_no_norms: both of the above;
* query_pairs_per_head_w16: per-head budgets over 16-token windows (decisions wait 16 tokens);
* query_pairs_per_head_padded: the same budget for every head, reserved in full every window;
* query_pairs_w16 and query_pairs_per_head_padded_w16: query_pairs and the padded
  per-head arm over 16-token windows;
* query_pairs_head_split: a fixed budget per head that differs by head, set once from
  each head's share under the shared window budget;
* query_pairs_head_levels: that split rounded to 0/12.5/25/37.5/49% per head, with
  each head's budget reserved (padded) in every window.

The official method (locked in 2026-10-18) is query_pairs_per_head_padded_w16:
pairs with query-aware keys, the same budget for every KV head reserved in full
(fixed-size blocks), and 16-token windows. ``--official`` runs it at
OFFICIAL_BUDGETS: Llama 3.1 8B-Instruct on C-Eval, GSM8K and HumanEval-instruct,
and ``--model codellama`` CodeLlama 7B on HumanEval with its body-indent prompt. Each model gets its own
query and layer calibration.

Every window of a layer slot is compressed to that slot's saving.
Pairs cannot reach 50%; their 50% point is the labeled endpoint where every pair
is merged with no residual (about 49.15%).
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.models import CODELLAMA_7B, LLAMA31_8B, QWEN35_9B
from compression_topics.spatial.algorithms.formats import max_saving
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json
from engine.layer_select.study import require_matched_savings, select_settings

SIMPLIFICATIONS = {
    "query_pairs_per_head": dict(per_head_budget=True),
    "query_pairs_no_rope": dict(align_rope=False),
    "query_pairs_no_norms": dict(keep_norms=False),
    "query_pairs_no_rope_no_norms": dict(align_rope=False, keep_norms=False),
    "query_pairs_per_head_w16": dict(per_head_budget=True, window=16),
    # Same budget for every head, each head's budget reserved in full every window.
    "query_pairs_per_head_padded": dict(per_head_budget=True, pad_heads=True),
    # 16-token windows: the shared budget, and the same budget per head reserved in full.
    "query_pairs_w16": dict(window=16),
    "query_pairs_per_head_padded_w16": dict(per_head_budget=True, pad_heads=True, window=16),
}
# Per-head budgets that differ by head: each head's fixed share is measured once
# (study.calibrate_head_savings) and stays the same for every window.
HEAD_SPLIT = "query_pairs_head_split"
# The same split rounded to a few levels, with each head's budget reserved in full.
HEAD_LEVELS = "query_pairs_head_levels"
HEAD_ARMS = (HEAD_SPLIT, HEAD_LEVELS)
OFFICIAL_ARM = "query_pairs_per_head_padded_w16"
OFFICIAL_BUDGETS = (0.1, 0.2, 0.3, 0.35, 0.4)
# Model: (model_args, official tasks, shape). Qwen3.5-9B is hybrid: only every
# fourth layer is full attention with a KV cache, and its heads hold 256 values.
MODELS = {
    "llama": (LLAMA31_8B, ("ceval", "gsm8k", "humaneval"), {}),
    "codellama": (CODELLAMA_7B, ("humaneval_codellama",), {}),
    "qwen": (QWEN35_9B, ("ceval", "gsm8k", "humaneval"),
             dict(head_dim=256, kv_layers=tuple(range(3, 32, 4)))),
}
OFFICIAL_TASKS = MODELS["llama"][1]
ARMS = ("pairs", "hierarchy", "query_pairs", "query_hierarchy", *SIMPLIFICATIONS, *HEAD_ARMS)
PAIR_ARMS = ("pairs", "query_pairs", *SIMPLIFICATIONS, *HEAD_ARMS)
QUERY_ARMS = ("query_pairs", "query_hierarchy", *SIMPLIFICATIONS, *HEAD_ARMS)
BUDGETS = (0.3, 0.4, 0.5)
OUT = ROOT / "compression_topics/spatial/figures/ladder"


def plan(out, *, arms=ARMS[:4], budgets=BUDGETS, layers=32, screen=4, refine=12, validation=16,
         tasks=("ceval",), model_args=LLAMA31_8B, head_dim=128, kv_layers=None):
    p = base.base_plan(
        out,
        model_args=model_args,
        head_dim=head_dim,
        kv_layers=kv_layers,
        layers=layers,
        budgets=budgets,
        screen=screen,
        refine=refine,
        validation=validation,
    )
    p.update(
        version=2,
        protocol="spatial_ladder_fixed_windows_v2",
        arms=list(arms),
        pair_local_budgets=[0.1, 0.25, 0.4, 0.47, 0.49],
        local_budgets=[0.1, 0.25, 0.4, 0.47, 0.49, 0.55, 0.7, 0.8, 0.85],
        query_weights=str(Path(out) / "query_weights.pt"),
        query_chunks=2,
        query_prefix=1536,
        pair_endpoint="every_pair_merged",
    )
    if tuple(tasks) != ("ceval",):
        p["tasks"] = list(tasks)
    if any(a in HEAD_ARMS for a in arms):
        p["head_savings"] = str(Path(out) / "head_savings.json")
    return p


def head_split_budgets(study):
    """Local budgets the head-split arm is calibrated at (its per-head shares are measured for each)."""
    return study["pair_local_budgets"] + ([0.5] if 0.5 in study["budgets"] else [])


def _config(study, arm, target, digest):
    aware = arm in QUERY_ARMS and target == "k"
    return SpatialConfig(
        max_group=2 if arm in PAIR_ARMS else 8,
        query_aware=aware,
        query_weights=study["query_weights"] if aware else None,
        query_weights_sha256=digest if aware else None,
        **SIMPLIFICATIONS.get(arm, {}),
    )


def candidates(study):
    query = any(a in QUERY_ARMS for a in study["arms"])
    path = Path(study["query_weights"])
    if query and not path.exists():
        raise ValueError("fresh query calibration required")
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if query else None
    shares = None
    if any(a in HEAD_ARMS for a in study["arms"]):
        source = Path(study["head_savings"])
        if not source.exists():
            raise ValueError("per-head budget calibration required")
        shares = json.loads(source.read_text())
    rows = []
    for arm in study["arms"]:
        pairs = arm in PAIR_ARMS
        endpoint = pairs and 0.5 in study["budgets"]
        budgets = study["pair_local_budgets"] if pairs else study["local_budgets"]
        for layer in base.kv_layers(study):
            for target in ("k", "v"):
                config = _config(study, arm, target, digest)
                for budget in budgets + ([0.5] if endpoint else []):
                    if arm in HEAD_ARMS:
                        table = shares["savings" if arm == HEAD_SPLIT else "levels"]
                        config = replace(config, head_savings=tuple(
                            table[str(layer)][target][str(budget)]), pad_heads=arm == HEAD_LEVELS)
                    rows.append(
                        dict(
                            name=f"{arm}_L{layer:02d}_{target}_b{round(100 * budget):02d}",
                            arm=arm,
                            layer=layer,
                            target=target,
                            budget=budget,
                            maximum_feasible=endpoint and budget == 0.5,
                            kv=config.pipeline(budget, **base.slot(layer, target)),
                        )
                    )
    return rows


def allocations(study, calibration):
    result = []
    for arm in study["arms"]:
        regular = [
            r
            for r in calibration["candidates"]
            if r["arm"] == arm and not r.get("maximum_feasible")
        ]
        budgets = [b for b in study["budgets"] if not (arm in PAIR_ARMS and b == 0.5)]
        for selection in base.allocate(
            study, dict(calibration, candidates=regular), arm, budgets
        ):
            selection.update(maximum_feasible=False)
            result.append(selection)
        if arm in PAIR_ARMS and 0.5 in study["budgets"]:
            settings = select_settings(
                calibration,
                include=lambda r, arm=arm: r["arm"] == arm and r.get("maximum_feasible"),
            )
            if len(settings) != len(base.kv_layers(study)) * 2:
                raise ValueError("missing pair endpoint calibration")
            pipeline = [step for r in settings for step in r["kv"]["pipeline"]]
            result.append(
                dict(
                    arm=arm,
                    budget=0.5,
                    tag=f"{arm}_b50_maximum_feasible",
                    maximum_feasible=True,
                    kv={"pipeline": pipeline},
                    assignment=settings,
                    compression=sum(r["measured_saving"] for r in settings)
                    / len(settings),
                    allocation_basis="measured_calibration_every_pair_merged",
                )
            )
    return result


def check_validation(rows, budgets, arms, *, tolerance):
    for budget in budgets:
        eligible = [a for a in arms if not (a in PAIR_ARMS and budget == 0.5)]
        if eligible:
            require_matched_savings(
                [r for r in rows if r["arm"] in eligible],
                [budget],
                eligible,
                tolerance=tolerance,
            )
    ceiling = max_saving(128)  # one mean, two norms and a tag per pair (about the same at 256)
    for arm in PAIR_ARMS:
        endpoint = [r for r in rows if r["arm"] == arm and r["budget"] == 0.5]
        if endpoint and (
            len(endpoint) != 1
            or not ceiling - tolerance <= endpoint[0]["actual_saving"] <= ceiling + 1e-8
        ):
            raise ValueError("pair endpoint differs from the physical byte ceiling")


def tasks(study, selections):
    run = base.task_run(study, selections)
    by_tag = {s["tag"]: s for s in selections}
    for config in run["configurations"]:
        selection = by_tag.get(config["name"].rsplit("_", 1)[0])
        if selection is None:
            continue
        config["metadata"].update(
            maximum_feasible=selection["maximum_feasible"],
            operating_point=(
                "maximum_feasible_pairs"
                if selection["maximum_feasible"]
                else "calibrated_budget"
            ),
        )
    return run


def summarize(out, run, *, require_complete=False):
    report = base.summarize(
        out,
        run,
        require_complete=require_complete,
        arm_pairs=(
            ("pairs", "hierarchy"),
            ("pairs", "query_pairs"),
            ("hierarchy", "query_hierarchy"),
            ("query_pairs", "query_hierarchy"),
        ),
        scope="spatial ladder; fixed eight-token windows; full C-Eval 5-shot",
        comparison="independent calibration per arm; pair 50% is the every-pair-merged endpoint",
    )
    for comparison in report["comparisons"]:
        if comparison["budget"] == 0.5 and comparison["reference_arm"] in PAIR_ARMS:
            comparison.update(
                matched_within_one_point=False, maximum_feasible_reference=True
            )
    write_json(
        Path(out)
        / ("summary.json" if report["status"] == "complete" else "preliminary.json"),
        report,
    )
    return report


def run_stage(study, stage):
    return base.run_stage(
        study,
        stage,
        candidates=candidates,
        arms=study["arms"],
        allocations=allocations,
        tasks=tasks,
        summarizer=summarize,
        validation_checker=check_validation,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS[:4]))
    parser.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    parser.add_argument("--from", dest="start", choices=base.STAGES, default="screen")
    parser.add_argument("--through", choices=base.STAGES, default="summary")
    parser.add_argument("--tasks", choices=tuple(base.TASKS), nargs="+", default=["ceval"])
    parser.add_argument("--model", choices=tuple(MODELS), default="llama")
    parser.add_argument("--official", action="store_true",
                        help="the official method at OFFICIAL_BUDGETS on the model's official tasks")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.official:
        args.arms, args.budgets = [OFFICIAL_ARM], list(OFFICIAL_BUDGETS)
        args.tasks = list(MODELS[args.model][1])
    study = plan(args.out.resolve(), arms=tuple(dict.fromkeys(args.arms)),
                 budgets=tuple(args.budgets), tasks=tuple(dict.fromkeys(args.tasks)),
                 model_args=MODELS[args.model][0], **MODELS[args.model][2])
    base.write_plan(study)
    print(
        f"Spatial ladder: {', '.join(study['arms'])} at "
        f"{'/'.join(str(round(100 * b)) for b in study['budgets'])}%; "
        f"4+12 train and 16 holdout chunks; tasks: {', '.join(study.get('tasks', ['ceval']))}.",
        flush=True,
    )
    if args.execute:
        if any(a in QUERY_ARMS for a in study["arms"]):
            base.calibrate_queries(study)
        if any(a in HEAD_ARMS for a in study["arms"]):
            base.calibrate_head_savings(study, head_split_budgets(study))
        base.run_stages(study, args.start, args.through, run_stage)


if __name__ == "__main__":
    main()
