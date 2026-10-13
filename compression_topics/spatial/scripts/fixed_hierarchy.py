#!/usr/bin/env python3
"""40% query-aware hierarchy with calibrated fixed residuals per group size."""

import argparse
import hashlib
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.algorithms.formats import MENUS, max_saving
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json

ARM = "fixed_hierarchy"
REFERENCE = (
    ROOT / "compression_topics/spatial/figures/clean_online_ladder_query_hierarchy"
)
OUT = ROOT / "compression_topics/spatial/figures/fixed_hierarchy_40"


def plan(out, *, layers=32):
    p = base.base_plan(
        out, layers=layers, budgets=(0.4,), screen=2, refine=12, validation=16
    )
    p.update(
        version=1,
        protocol="query_fixed_group_residuals_prefill_only_v1",
        arms=[ARM],
        local_budgets=[0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85],
        query_weights=str(REFERENCE / "query_weights.pt"),
        query_weights_sha256=hashlib.sha256(
            (REFERENCE / "query_weights.pt").read_bytes()
        ).hexdigest(),
        residual_combinations=[list(r) for r in product(MENUS[2], MENUS[4], MENUS[8])],
        reference=str(REFERENCE),
        holdout_ppl_ratio_limit=1.02,
        accuracy_tolerance_points=1.0,
        prefill_only=True,
    )
    return p


def candidates(p):
    rows = []
    for layer in range(p["layers"]):
        for target in ("k", "v"):
            for combo in p["residual_combinations"]:
                counts = dict(zip(("2", "4", "8"), combo))
                ceiling = max(
                    max_saving(p["head_dim"], g, counts[str(g)]) for g in MENUS
                )
                aware = target == "k"
                config = SpatialConfig(
                    max_group=8,
                    residual_counts=counts,
                    prefill_only=True,
                    control=p["control"],
                    query_aware=aware,
                    query_weights=p["query_weights"] if aware else None,
                    query_weights_sha256=p["query_weights_sha256"] if aware else None,
                )
                for budget in p["local_budgets"]:
                    if budget >= ceiling:
                        continue
                    rows.append(
                        dict(
                            name=f"fixed_L{layer:02d}_{target}_b{round(100*budget):02d}_r"
                            + "_".join(map(str, combo)),
                            arm=ARM,
                            layer=layer,
                            target=target,
                            budget=budget,
                            residual_counts=counts,
                            kv=config.pipeline(budget, **base.slot(layer, target)),
                        )
                    )
    return rows


from compression_topics.spatial.scripts.comparison import (
    reference_selection,
    quality_gate,
    run_comparison_stage,
)
from compression_topics.spatial.scripts import comparison


def allocations(p, cal):
    return base.allocate(p, cal, ARM)


def tasks(p, selections):
    quality_gate(p)
    return base.task_run(p, selections)


def summarize(out, run, *, require_complete=False, arm=ARM):
    return comparison.summarize(out, run, require_complete=require_complete, arm=arm)


def run_stage(p, stage):
    return run_comparison_stage(
        p,
        stage,
        candidates=candidates,
        allocations=allocations,
        tasks=tasks,
        summarizer=summarize,
        reference_builder=reference_selection,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--from", dest="start", choices=base.STAGES, default="screen")
    parser.add_argument("--through", choices=base.STAGES, default="summary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    p = plan(args.out.resolve())
    base.write_plan(p)
    print(
        f"{len(candidates(p))} screening candidates; 12 residual combinations; one 40% C-Eval point after holdout gate.",
        flush=True,
    )
    if args.execute:
        base.run_stages(p, args.start, args.through, run_stage)


if __name__ == "__main__":
    main()
