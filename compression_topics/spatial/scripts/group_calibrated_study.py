#!/usr/bin/env python3
"""Calibrate shared-residual pairs versus quads per independent layer/K/V slot.

Pairs screen zero residual and the largest feasible residual size; quads screen the full grid.
Screen residual sizes separately for each group size, then refine the best of
EACH size on identical disjoint chunks. Close/reordered within-size finalists
and close pair-versus-quad comparisons receive more chunks. Final tasks compare
pairs-only against a per-slot pair/quad policy. Each assigned slot uses one
fixed group size; individual groups can remain dense. Prefill only.
All calibration and task budgets include norms, residual masks, byte-packed
merge flags for merged/dense groups, and a two-byte slot header. Values/norms
are accounted at 16 bits. This remains a reconstruction simulation, not a
measurement of allocated PyTorch cache memory.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms.group_calibrated import ACCOUNTING, group_saving
from engine.eval_runner.files import write_json
from engine.layer_select.calibration import (configuration, measurement, perplexity,
    subset_manifest, execute_chunks, run_stages, STAGES)
from engine.layer_select.greedy.calibrated import allocate
from engine.layer_select.calibration import select as base_select, comparisons as base_comparisons
GROUP_FIELDS = ("layer", "target", "budget", "group_size")


GROUP_SIZES = (2, 4)


def residual_choices(budget, residuals, group_size, dim=128):
    feasible = [k for k in sorted(set(residuals))
                if budget <= group_saving(k, dim=dim, group_size=group_size) + 1e-12]
    if group_size == 2 and feasible:
        return sorted({feasible[0], feasible[-1]})
    return feasible

LAYERS = 32
GLOBAL_BUDGETS = (.1, .2, .3, .4, .5, .6)
LOCAL_BUDGETS = (.1, .2, .25, .3, .4, .49, .5, .6, .65, .7, .74)
RESIDUAL_ENTRIES = (0, 4, 8, 16, 32)


def step(layer, target, budget, kept, group_size):
    return {"method": "group_calibrated", "accounting": ACCOUNTING, "group_size": group_size, "k_layers": [layer] if target == "k" else [],
            "v_layers": [layer] if target == "v" else [], "saving": budget,
            "residual_entries": kept}


def plan(out, layers=LAYERS, local_budgets=LOCAL_BUDGETS, residuals=RESIDUAL_ENTRIES,
         screen_chunks=16, refine_chunks=32, extra_chunks=32, seed=0, seq_len=2048, model_args=LLAMA31_8B, head_dim=128, finalists=1):
    task = dict(WIKITEXT_FULL)
    configs = [configuration("dense_wikitext", {"pipeline": []}, out / "sweep", task)]
    candidates = []
    for layer in range(layers):
        for target in ("k", "v"):
            for budget in local_budgets:
                for group_size in GROUP_SIZES:
                    for kept in residual_choices(budget, residuals, group_size, dim=head_dim):
                        maximum = group_saving(kept, dim=head_dim, group_size=group_size)
                        if budget > maximum + 1e-12:
                            continue
                        name = f"g{group_size}_L{layer:02d}_{target}_b{round(budget * 100):02d}_r{kept:02d}"
                        kv = {"pipeline": [step(layer, target, budget, kept, group_size)]}
                        configs.append(configuration(name, kv, out / "sweep", task))
                        candidates.append({"name": name, "layer": layer, "target": target,
                            "budget": budget, "group_size": group_size, "residual_entries": kept, "kv": kv,
                            "nominal_merge_fraction": budget / maximum,
                            "output_path": configs[-1]["output_path"]})
    return {"layers": layers, "head_dim": head_dim, "local_budgets": list(local_budgets), "residual_entries": list(residuals),
            "global_budgets": list(GLOBAL_BUDGETS), "screen_chunks": screen_chunks,
            "refine_chunks": refine_chunks, "extra_chunks": extra_chunks, "seed": seed,
            "seq_len": seq_len, "finalists": finalists,
            "norm_overhead_counted": True, "flags_counted": True, "residual_mask_counted": True, "slot_header_counted": True, "norm_bits_per_token": 16, "merge_flag_bits_per_group": 1, "slot_header_bits": 16, "group_sizes": list(GROUP_SIZES), "accounting": ACCOUNTING,
            "pair_residual_menu": "smallest and largest feasible residual counts",
            "pair_residual_representation": "one shared signed half-difference and mask",
            "ranking": "minimum reconstruction cosine across all group tokens",
            "candidates": candidates, "run": {"model": "hf", "model_args": model_args,
                "batch_size": 1, "configurations": configs,
                "model_shape": {"layers": layers, "head_dim": head_dim},
                "sampling": {"pool_chunks": screen_chunks + refine_chunks + extra_chunks,
                             "offset": 0, "chunks": screen_chunks, "seed": seed, "seq_len": seq_len}}}


def comparisons(calibration, screening, trigger_changed=True):
    reports = base_comparisons(calibration, screening, trigger_changed, group_fields=GROUP_FIELDS)
    slots = sorted({(r["layer"], r["target"], r["budget"]) for r in calibration["selected"]})
    for slot in slots:
        choices = sorted([r for r in calibration["selected"] if
                          (r["layer"], r["target"], r["budget"]) == slot], key=lambda r: r["ppl"])
        if len(choices) < 2:
            continue
        a = {r["chunk"]: r["nll"] for r in choices[0]["chunk_scores"]}
        b = {r["chunk"]: r["nll"] for r in choices[1]["chunk_scores"]}
        if set(a) != set(b) or len(a) < 2:
            raise ValueError("pair/quad comparison needs identical chunk IDs")
        differences = [b[i] - a[i] for i in sorted(a)]
        gap = statistics.mean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences))
        near = gap <= 1.96 * se
        for report in reports:
            if (report["layer"], report["target"], report["budget"]) == slot:
                report.update(cross_size_near_tie=near, cross_size_gap=gap,
                              cross_size_standard_error=se)
                report["needs_more"] |= near
    return reports


def policy_calibration(calibration, policy):
    rows = [r for r in calibration["selected"] if policy == "mixed" or r["group_size"] == 2]
    slots = sorted({(r["layer"], r["target"], r["budget"]) for r in rows})
    winners = [min((r for r in rows if (r["layer"], r["target"], r["budget"]) == slot),
                   key=lambda r: (r["ppl"], r["group_size"], r["residual_entries"])) for slot in slots]
    return {**calibration, "selected": winners}


def task_plan(calibration, layers, out):
    configs = [configuration("dense_ceval", {"pipeline": []}, out / "tasks", CEVAL_VALID_5SHOT)]
    selections = []
    for policy in ("pairs", "mixed"):
        calibrated = policy_calibration(calibration, policy)
        for mode, targets in (("keys", ("k",)), ("kv", ("k", "v"))):
            for budget in GLOBAL_BUDGETS:
                if policy == "pairs" and budget > group_saving(0, group_size=2):
                    continue
                selection = allocate(calibrated, layers, targets, budget)
                selection.update(mode=mode, policy=policy, tag=f"{policy}_{mode}_b{round(budget * 100):02d}")
                selections.append(selection)
                configs.append(configuration(selection["tag"], selection["kv"], out / "tasks", CEVAL_VALID_5SHOT,
                    kv_budget=budget, kv_compression=selection["compression"], compression_target="k" if mode == "keys" else "kv", policy=policy))
    return selections, {"model": "hf", "model_args": LLAMA31_8B, "batch_size": 1, "configurations": configs}


def execute(path, out):
    subprocess.run([sys.executable, str(ROOT / "engine/multi_run.py"), "--run", str(path),
                    "--skip-existing", "--write-results", "--results-root", str(ROOT)], cwd=ROOT, check=True)


def summary(out, selections):
    dense = json.loads((out / "tasks/dense_ceval.json").read_text())["results"]["ceval-valid"]["acc,none"]
    rows = []
    for selection in selections:
        payload = json.loads((out / "tasks" / f"{selection['tag']}.json").read_text())
        measured = measurement(payload)
        # Measurement counts only compressed slots. Include dense slots using
        # equal tensor sizes per layer, as in this model's uniform KV layout.
        selected_slots = len(selection["assignment"])
        per_slot_saving = measured["compression"]
        # The study manifest gives the exact number of layers (not the sparse assignment).
        layers = json.loads((out / "manifest.json").read_text())["layers"]
        denominator = layers if selection["mode"] == "keys" else 2 * layers
        storage = payload.get("storage", {}).get("targets", {})
        scope = "k" if selection["mode"] == "keys" else "kv"
        measured_saving = storage.get(scope, {}).get("compression", per_slot_saving * selected_slots / denominator)
        whole_saving = storage.get("kv", {}).get("compression", per_slot_saving * selected_slots / (2 * layers))
        accuracy = payload["results"]["ceval-valid"]["acc,none"]
        rows.append({"policy": selection["policy"], "group_size_counts": {str(g): sum(r["group_size"] == g for r in selection["assignment"]) for g in GROUP_SIZES}, "mode": selection["mode"], "budget": selection["budget"],
            "planned_compression": selection["compression"],
            "measured_compression": measured_saving,
            "whole_kv_compression": whole_saving,
            "accuracy": accuracy, "delta_accuracy": accuracy - dense, "cosine": measured})
    return {"dense_accuracy": dense, "task": "ceval-valid", "shots": 5, "group_sizes": list(GROUP_SIZES), "accounting": ACCOUNTING, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--local-budgets", type=float, nargs="+", default=list(LOCAL_BUDGETS))
    parser.add_argument("--residual-entries", type=int, nargs="+", default=list(RESIDUAL_ENTRIES))
    parser.add_argument("--screen-chunks", type=int, default=16)
    parser.add_argument("--refine-chunks", type=int, default=32)
    parser.add_argument("--extra-chunks", type=int, default=32, help="additional chunks for unstable or close finalist comparisons; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--finalists", type=int, default=1)
    parser.add_argument("--from", dest="start", choices=STAGES, default="sweep")
    parser.add_argument("--through", choices=STAGES, default="summary")
    parser.add_argument("--execute", action="store_true", help="explicitly start evaluations; otherwise only write the plan")
    parser.add_argument("--model-args", default=LLAMA31_8B)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if args.layers < 1 or args.head_dim < 1:
        parser.error("layers and head-dim must be positive")
    if args.out is None:
        args.out = ROOT / "compression_topics/spatial/figures/pair_quad_shared_sampled"
    if STAGES.index(args.start) > STAGES.index(args.through):
        parser.error("--from must not follow --through")
    budgets = sorted(set(args.local_budgets))
    if not budgets or any(not 0 < b <= .75 for b in budgets) or any(abs(b * 100 - round(b * 100)) > 1e-9 for b in budgets):
        parser.error("local budgets must be whole percentages in (0, 75%]")
    if args.screen_chunks < 1 or args.refine_chunks < 2 or args.extra_chunks < 0 or args.finalists < 1 or args.seq_len < 2:
        parser.error("need positive screening/finalist counts, at least two refinement chunks, nonnegative extra chunks and seq-len >= 2")
    residuals = sorted(set(args.residual_entries))
    if not residuals or any(not 0 <= r <= 32 for r in residuals):
        parser.error("residual entries must be between 0 and 32")
    manifest = plan(args.out.resolve(), layers=args.layers, model_args=args.model_args, head_dim=args.head_dim, local_budgets=budgets, residuals=residuals,
                    screen_chunks=args.screen_chunks, refine_chunks=args.refine_chunks,
                    extra_chunks=args.extra_chunks, seed=args.seed, seq_len=args.seq_len, finalists=args.finalists)
    if any(not any(c["budget"] == b for c in manifest["candidates"]) for b in budgets):
        parser.error("a local budget has no feasible residual candidate")
    args.out = args.out.resolve()
    existing_path = args.out / "manifest.json"
    if existing_path.exists():
        previous = json.loads(existing_path.read_text())
        fields = ("layers", "local_budgets", "residual_entries", "screen_chunks", "refine_chunks",
                  "extra_chunks", "seed", "seq_len", "finalists", "ranking", "group_sizes", "global_budgets", "accounting")
        if (previous["run"]["model_args"] != manifest["run"]["model_args"] or
                previous.get("head_dim", 128) != manifest["head_dim"] or
                any(previous.get(key) != manifest.get(key) for key in fields)):
            parser.error("existing output directory has a different calibration grid; use a new --out directory")
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "sweep_run.json", manifest["run"])
    print(f"Plan: {len(manifest['candidates'])} candidates on {args.screen_chunks} random chunks; "
          f"up to {args.finalists} finalists per group size/slot/budget on {args.refine_chunks} additional chunks.")
    print(f"Unstable/close comparisons get {args.extra_chunks} extra chunks; final C-Eval: pairs-only and mixed pair/quad policies + dense, full dataset.")
    print(f"Residual entries: {residuals}; local budgets: {budgets}; global budgets: {GLOBAL_BUDGETS}")
    print(f"Output: {args.out}")
    if not args.execute:
        print("Plan only. Add --execute when ready to run.")
        return
    run_stages(args, manifest, select=select, comparisons=comparisons,
               task_plan=task_plan, execute=execute, summary=summary)


def select(manifest, tolerance=.001, top_k=1):
    return base_select(manifest, tolerance, top_k, group_fields=GROUP_FIELDS)


if __name__ == "__main__":
    main()
