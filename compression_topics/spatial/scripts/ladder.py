#!/usr/bin/env python3
"""The official spatial pair compression sweep.

Method (locked in 2026-10-26): pairs with query-aware keys, the same budget for
every KV head in fixed-size padded blocks, 16-token windows, vectors merged as
stored in the cache. Per-layer budgets come from the greedy allocation over
single-slot calibrations (study.py).

Runs the given model at BUDGETS (default 10/20/30/35/40%) on its tasks:
Llama 3.1 8B-Instruct and Qwen3.5-9B on C-Eval, GSM8K and HumanEval-instruct,
CodeLlama 7B on HumanEval with its body-indent prompt. Each model gets its own
query and layer calibration.
"""

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.models import CODELLAMA_7B, LLAMA31_8B, QWEN35_9B
from compression_topics.spatial.algorithms.methods import SpatialConfig
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json

ARM = "query_pairs_per_head_padded_w16"
BUDGETS = (0.1, 0.2, 0.3, 0.35, 0.4)
LOCAL_BUDGETS = (0.1, 0.25, 0.4, 0.47, 0.49)  # single-slot calibration points
# Model: (model_args, tasks, shape). Qwen3.5-9B is hybrid: only every fourth layer
# is full attention with a KV cache, and its heads hold 256 values.
MODELS = {
    "llama": (LLAMA31_8B, ("ceval", "gsm8k", "humaneval"), {}),
    "codellama": (CODELLAMA_7B, ("humaneval_codellama",), {}),
    "qwen": (QWEN35_9B, ("ceval", "gsm8k", "humaneval"),
             dict(head_dim=256, kv_layers=tuple(range(3, 32, 4)))),
}
OUT = ROOT / "compression_topics/spatial/figures/official_llama"


def plan(out, *, budgets=BUDGETS, layers=32, screen=4, refine=12, validation=16,
         tasks=MODELS["llama"][1], model_args=LLAMA31_8B, head_dim=128, kv_layers=None):
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
        version=3,
        protocol="spatial_pairs_official_v3",
        arms=[ARM],
        local_budgets=list(LOCAL_BUDGETS),
        query_weights=str(Path(out) / "query_weights.pt"),
        query_chunks=2,
        query_prefix=1536,
        tasks=list(tasks),
    )
    return p


def candidates(study):
    """One candidate per KV layer, K or V, and local budget: only that slot is compressed."""
    path = Path(study["query_weights"])
    if not path.exists():
        raise ValueError("fresh query calibration required")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = SpatialConfig(query_weights=study["query_weights"], query_weights_sha256=digest)
    rows = []
    for layer in base.kv_layers(study):
        for target in ("k", "v"):
            config = keys if target == "k" else SpatialConfig()
            for budget in study["local_budgets"]:
                rows.append(dict(
                    name=f"{ARM}_L{layer:02d}_{target}_b{round(100 * budget):02d}",
                    arm=ARM,
                    layer=layer,
                    target=target,
                    budget=budget,
                    kv=config.pipeline(budget, **base.slot(layer, target)),
                ))
    return rows


def allocations(study, calibration):
    return base.allocate(study, calibration, ARM)


def summarize(out, run, *, require_complete=False):
    return base.summarize(out, run, require_complete=require_complete,
                          scope="official spatial pair compression")


def run_stage(study, stage):
    return base.run_stage(
        study,
        stage,
        candidates=candidates,
        arms=study["arms"],
        allocations=allocations,
        tasks=base.task_run,
        summarizer=summarize,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=tuple(MODELS), default="llama")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default figures/official_<model>)")
    parser.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    parser.add_argument("--tasks", choices=tuple(base.TASKS), nargs="+", default=None,
                        help="default: the model's tasks")
    parser.add_argument("--from", dest="start", choices=base.STAGES, default="screen")
    parser.add_argument("--through", choices=base.STAGES, default="summary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model_args, model_tasks, shape = MODELS[args.model]
    out = args.out or OUT.with_name(f"official_{args.model}")
    study = plan(out.resolve(), budgets=tuple(args.budgets),
                 tasks=tuple(dict.fromkeys(args.tasks or model_tasks)),
                 model_args=model_args, **shape)
    base.write_plan(study)
    print(
        f"Spatial pair compression on {args.model} at "
        f"{'/'.join(str(round(100 * b)) for b in study['budgets'])}%; "
        f"4+12 train and 16 holdout chunks; tasks: {', '.join(study['tasks'])}.",
        flush=True,
    )
    if args.execute:
        base.calibrate_queries(study)
        base.run_stages(study, args.start, args.through, run_stage)


if __name__ == "__main__":
    main()
