#!/usr/bin/env python3
"""Spatial ladder at 30/40/50%: pairs, hierarchy, query-aware hierarchy.

Arms (each calibrated and allocated independently):
* pairs: dense or pairs with 0/16/32 residual entries;
* hierarchy: the same pairs plus quads and eights with 0/16 entries;
* query_hierarchy: the hierarchy with query-aware loss and residuals on keys;
* query_pairs: the pairs with query-aware loss and residuals on keys.

``--algorithm online`` (default) steers the price causally and also compresses
generated tokens; ``--algorithm chunked`` compresses every eight-token window to
the slot's saving on its own, also during generation; ``--algorithm offline``
fits one price per slot over the prompt. Pairs cannot reach 50%; online and
chunked, its 50% point is the labeled minimum-storage endpoint (every pair
merged, no residual, about 49.15%); offline skips it.
"""

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms.formats import max_saving
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json
from engine.layer_select.study import require_matched_savings, select_settings

ARMS = ("pairs", "hierarchy", "query_hierarchy", "query_pairs")
PAIR_ARMS = ("pairs", "query_pairs")
QUERY_ARMS = ("query_hierarchy", "query_pairs")
BUDGETS = (0.3, 0.4, 0.5)
OUT = ROOT / "compression_topics/spatial/figures/clean_ladder"


def plan(
    out, *, arms=ARMS, algorithm="online", layers=32, screen=4, refine=12, validation=16
):
    p = base.base_plan(
        out,
        layers=layers,
        budgets=BUDGETS,
        screen=screen,
        refine=refine,
        validation=validation,
    )
    p.update(
        version=1,
        protocol="spatial_ladder_offline_online_v1",
        algorithm=algorithm,
        arms=list(arms),
        pair_local_budgets=[0.1, 0.25, 0.4, 0.47, 0.49],
        local_budgets=[0.1, 0.25, 0.4, 0.47, 0.49, 0.55, 0.7, 0.8, 0.85],
        query_weights=str(Path(out) / "query_weights.pt"),
        query_chunks=2,
        query_prefix=1536,
        pair_endpoint="minimum_storage_online_only",
    )
    if algorithm == "chunked":
        p["pair_endpoint"] = "minimum_storage_chunked"
    if algorithm != "online":
        p.pop("control")
    return p


def _config(study, arm, target, digest, minimum_storage=False):
    aware = arm in QUERY_ARMS and target == "k"
    return SpatialConfig(
        algorithm=study["algorithm"],
        max_group=2 if arm in PAIR_ARMS else 8,
        query_aware=aware,
        query_weights=study["query_weights"] if aware else None,
        query_weights_sha256=digest if aware else None,
        control=study.get("control"),
        minimum_storage=minimum_storage,
    )


def candidates(study):
    query = any(a in QUERY_ARMS for a in study["arms"])
    path = Path(study["query_weights"])
    if query and not path.exists():
        raise ValueError("fresh query calibration required")
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if query else None
    rows = []
    for arm in study["arms"]:
        budgets = (
            study["pair_local_budgets"] if arm in PAIR_ARMS else study["local_budgets"]
        )
        endpoint = arm in PAIR_ARMS and study["algorithm"] != "offline"
        for layer in range(study["layers"]):
            for target in ("k", "v"):
                for budget in budgets + ([0.5] if endpoint else []):
                    minimum = endpoint and budget == 0.5
                    config = _config(study, arm, target, digest, minimum)
                    rows.append(
                        dict(
                            name=f"{arm}_L{layer:02d}_{target}_b{round(100 * budget):02d}",
                            arm=arm,
                            layer=layer,
                            target=target,
                            budget=budget,
                            maximum_feasible=minimum,
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
        budgets = [0.3, 0.4] if arm in PAIR_ARMS else study["budgets"]
        for selection in base.allocate(
            study, dict(calibration, candidates=regular), arm, budgets
        ):
            selection.update(maximum_feasible=False)
            result.append(selection)
        if arm in PAIR_ARMS and study["algorithm"] != "offline":
            settings = select_settings(
                calibration,
                include=lambda r, arm=arm: r["arm"] == arm and r.get("maximum_feasible"),
            )
            if len(settings) != study["layers"] * 2:
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
                    allocation_basis="measured_calibration_minimum_storage_endpoint",
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
    ceiling = max_saving(128)  # one mean, two norms and a tag per pair; slightly lower with headers
    for arm in PAIR_ARMS:
        endpoint = [r for r in rows if r["arm"] == arm and r["budget"] == 0.5]
        if endpoint and (
            len(endpoint) != 1
            or not ceiling - tolerance <= endpoint[0]["actual_saving"] <= ceiling + 1e-8
        ):
            raise ValueError("pair endpoint differs from the physical byte ceiling")


def tasks(study, selections):
    run = base.task_run(study, selections)
    for config, selection in zip(run["configurations"][1:], selections):
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
            ("hierarchy", "query_hierarchy"),
            ("pairs", "query_pairs"),
            ("query_pairs", "query_hierarchy"),
        ),
        scope="spatial ladder; per-head decisions; full C-Eval 5-shot",
        comparison="independent calibration per arm; pair 50% is a minimum-storage endpoint",
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
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--algorithm", choices=("online", "chunked", "offline"), default="online")
    parser.add_argument("--from", dest="start", choices=base.STAGES, default="screen")
    parser.add_argument("--through", choices=base.STAGES, default="summary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    study = plan(
        args.out.resolve(),
        arms=tuple(dict.fromkeys(args.arms)),
        algorithm=args.algorithm,
    )
    base.write_plan(study)
    print(
        f"Spatial ladder ({args.algorithm}): {', '.join(study['arms'])} at 30/40/50%; "
        "4+12 train and 16 holdout chunks, full C-Eval.",
        flush=True,
    )
    if args.execute:
        if any(a in QUERY_ARMS for a in study["arms"]):
            base.calibrate_queries(study)
        base.run_stages(study, args.start, args.through, run_stage)


if __name__ == "__main__":
    main()
