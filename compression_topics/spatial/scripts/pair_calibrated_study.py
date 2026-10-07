#!/usr/bin/env python3
"""Independent K/V slot calibration, followed by greedy allocation and C-Eval.

Default invocation only writes a plan; --execute explicitly starts evaluations.
Each (layer, target, byte budget) screens all feasible residual sizes on 16
seeded random 2,048-token chunks. The best two are retested on 32 disjoint chunks;
close or reordered finalists get another 32 chunks. Per-chunk losses and paired
comparison diagnostics are saved. This uses token PPL, not lm-eval word PPL.
Its merge
fraction is set from the byte budget, and pairs rank by actual reconstruction
cosine. The best measured WikiText perplexity wins. Keys and values calibrate
separately, never against a shared layer budget.

Global keys-only budgets are fractions of key bytes; combined budgets are
fractions of KV bytes. Values remain dense in the keys-only study. Norm storage
and per-pair flags are excluded; residual masks are included. Prefill only,
batch size one. The local search finds the best candidate in the supplied grid,
not a continuous optimum. Greedy uses singleton PPL effects, which need not add
across layers; C-Eval measures the final multi-layer assignments.
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
from compression_topics.spatial.algorithms.pair_calibrated import pair_saving
from engine.layer_select.scores import word_perplexity

LAYERS = 32
GLOBAL_BUDGETS = (.1, .2, .3, .4)
LOCAL_BUDGETS = (.1, .2, .25, .3, .4, .5)
RESIDUAL_ENTRIES = (0, 4, 8, 16, 32)
STAGES = ("sweep", "shortlist", "refine", "stabilize", "select", "greedy", "tasks", "summary")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def step(layer, target, budget, kept):
    return {"method": "pair_calibrated", "k_layers": [layer] if target == "k" else [],
            "v_layers": [layer] if target == "v" else [], "saving": budget,
            "residual_entries": kept}


def configuration(name, kv, out, task, **metadata):
    return {"name": name, "kv": kv, "output_path": str(out / f"{name}.json"),
            "metadata": metadata,
            **{k: v for k, v in task.items() if k not in {"name_task", "file"}}}


def plan(out, layers=LAYERS, local_budgets=LOCAL_BUDGETS, residuals=RESIDUAL_ENTRIES,
         screen_chunks=16, refine_chunks=32, extra_chunks=32, seed=0, seq_len=2048, finalists=2):
    task = dict(WIKITEXT_FULL)
    configs = [configuration("dense_wikitext", {"pipeline": []}, out / "sweep", task)]
    candidates = []
    for layer in range(layers):
        for target in ("k", "v"):
            for budget in local_budgets:
                for kept in residuals:
                    maximum = pair_saving(kept)
                    if budget > maximum + 1e-12:
                        continue
                    name = f"L{layer:02d}_{target}_b{round(budget * 100):02d}_r{kept:02d}"
                    kv = {"pipeline": [step(layer, target, budget, kept)]}
                    configs.append(configuration(name, kv, out / "sweep", task))
                    candidates.append({"name": name, "layer": layer, "target": target,
                        "budget": budget, "residual_entries": kept, "kv": kv,
                        "nominal_merge_fraction": budget / maximum,
                        "output_path": configs[-1]["output_path"]})
    return {"layers": layers, "local_budgets": list(local_budgets), "residual_entries": list(residuals),
            "global_budgets": list(GLOBAL_BUDGETS), "screen_chunks": screen_chunks,
            "refine_chunks": refine_chunks, "extra_chunks": extra_chunks, "seed": seed,
            "seq_len": seq_len, "finalists": finalists,
            "norm_overhead_counted": False, "flags_counted": False, "residual_mask_counted": True,
            "ranking": "minimum of the two token reconstruction cosines",
            "candidates": candidates, "run": {"model": "hf", "model_args": LLAMA31_8B,
                "batch_size": 1, "configurations": configs,
                "sampling": {"pool_chunks": screen_chunks + refine_chunks + extra_chunks,
                             "offset": 0, "chunks": screen_chunks, "seed": seed, "seq_len": seq_len}}}


def measurement(payload):
    stats = payload.get("gate_stats") or {}
    rows = [v for k, v in stats.items() if "_layer_" in k]
    dense = sum(r["dense_bits"] for r in rows)
    if not dense:
        raise ValueError("missing slot byte measurements")
    pairs, merged = sum(r["pairs"] for r in rows), sum(r["merged"] for r in rows)
    return {"compression": 1 - sum(r["stored_bits"] for r in rows) / dense,
            "merged_fraction": merged / pairs if pairs else 0.,
            "min_reconstruction_cosine": min(r["cosine_min"] for r in rows),
            "mean_reconstruction_cosine": sum(r["cosine_sum"] for r in rows) / merged if merged else None,
            "mean_sequence_cutoff": sum(r["cutoff_sum"] for r in rows) / sum(r["updates"] for r in rows),
            "shortfall_updates": sum(r["shortfall_updates"] for r in rows),
            "stats": stats}


def perplexity(payload):
    metrics = payload.get("results", {}).get("wikitext_chunks", {})
    return float(metrics["token_perplexity,none"]) if metrics else word_perplexity(payload)


def select(manifest, tolerance=.001, top_k=1):
    dense_path = Path(manifest["run"]["configurations"][0]["output_path"])
    dense_ppl = perplexity(json.loads(dense_path.read_text()))
    measured = []
    winners = []
    shortlisted = []
    for candidate in manifest["candidates"]:
        path = Path(candidate["output_path"])
        if not path.exists():
            raise ValueError(f"unfinished calibration: {path}")
        payload = json.loads(path.read_text())
        if (payload.get("simulation") or {}).get("kv") != candidate["kv"]:
            raise ValueError(f"result does not match planned candidate: {path}")
        ppl = perplexity(payload)
        measured.append({**candidate, "ppl": ppl, "delta_ppl": ppl - dense_ppl,
                         "delta_nll": math.log(ppl) - math.log(dense_ppl),
                         **measurement(payload), "chunk_scores": payload.get("chunk_scores", [])})
    groups = sorted({(r["layer"], r["target"], r["budget"]) for r in manifest["candidates"]})
    for layer, target, budget in groups:
        choices = [r for r in measured if r["layer"] == layer and r["target"] == target
                   and r["budget"] == budget and r["compression"] >= budget - tolerance]
        if not choices:
            raise ValueError(f"no candidate meets L{layer} {target} budget {budget}")
        ranked = sorted(choices, key=lambda r: (r["ppl"], r["residual_entries"], r["name"]))
        winners.append(ranked[0])
        shortlisted.extend(ranked[:top_k])
    return {"dense_ppl": dense_ppl, "budget_tolerance": tolerance,
            "candidates": measured, "selected": winners, "shortlisted": shortlisted}


def subset_manifest(original, candidates, out, offset, chunks, reuse_from=False):
    by_name = {r["name"]: r for r in original["candidates"]}
    selected = [{**by_name[r["name"]], "output_path": str(out / f"{r['name']}.json")} for r in candidates]
    configs = [configuration("dense_wikitext", {"pipeline": []}, out, WIKITEXT_FULL)]
    configs.extend(configuration(r["name"], r["kv"], out, WIKITEXT_FULL) for r in selected)
    if reuse_from:
        for config, candidate in zip(configs[1:], candidates):
            config["reuse_path"] = candidate["output_path"]
        # Reuse dense on the same 32 refinement chunks as well.
        configs[0]["reuse_path"] = str(Path(candidates[0]["output_path"]).parent / "dense_wikitext.json")
    sampling = {**original["run"]["sampling"], "offset": offset, "chunks": chunks}
    return {**original, "candidates": selected,
            "run": {**original["run"], "configurations": configs, "sampling": sampling}}


def comparisons(calibration, screening, trigger_changed=True):
    screened = {(r["layer"], r["target"], r["budget"]): r["name"] for r in screening["selected"]}
    reports = []
    for winner in calibration["selected"]:
        group = (winner["layer"], winner["target"], winner["budget"])
        choices = sorted([r for r in calibration["candidates"] if
                          (r["layer"], r["target"], r["budget"]) == group and
                          r["compression"] >= r["budget"] - calibration["budget_tolerance"]],
                         key=lambda r: r["ppl"])
        changed = screened[group] != winner["name"]
        report = {"layer": group[0], "target": group[1], "budget": group[2],
                  "winner": winner["name"], "ordering_changed": changed,
                  "near_tie": False, "needs_more": trigger_changed and changed}
        if len(choices) > 1:
            a = {r["chunk"]: r["nll"] for r in choices[0]["chunk_scores"]}
            b = {r["chunk"]: r["nll"] for r in choices[1]["chunk_scores"]}
            if set(a) != set(b) or len(a) < 2:
                raise ValueError("finalist comparison requires matching chunk IDs and at least two chunks")
            differences = [b[i] - a[i] for i in sorted(a)]
            gap = statistics.mean(differences)
            se = statistics.stdev(differences) / math.sqrt(len(differences))
            near_tie = gap <= 1.96 * se
            report.update(runner_up=choices[1]["name"], chunks=len(differences), paired_nll_gap=gap,
                          paired_nll_standard_error=se, near_tie=near_tie,
                          needs_more=near_tie or (trigger_changed and changed))
        reports.append(report)
    return reports


def execute_chunks(path):
    subprocess.run([sys.executable, str(ROOT / "compression_topics/spatial/scripts/pair_calibration_chunks.py"),
                    "--run", str(path)], cwd=ROOT, check=True)


def allocate(calibration, layers, targets, budget):
    slots = [(layer, target) for layer in range(layers) for target in targets]
    choices = {slot: sorted([r for r in calibration["selected"]
                            if (r["layer"], r["target"]) == slot], key=lambda r: r["budget"]) for slot in slots}
    indices = {slot: -1 for slot in slots}
    total = 0.
    while total / len(slots) < budget - 1e-12:
        options = []
        for slot in slots:
            current_idx = indices[slot]
            next_idx = current_idx + 1
            if next_idx >= len(choices[slot]):
                continue
            current = None if current_idx == -1 else choices[slot][current_idx]
            nxt = choices[slot][next_idx]
            gain = nxt["compression"] - (current["compression"] if current else 0.)
            if gain <= 0:
                raise ValueError("calibrated rung compression must increase")
            cost = nxt["ppl"] - (current["ppl"] if current else calibration["dense_ppl"])
            efficiency = float("inf") if cost <= 0 else gain / cost
            options.append((efficiency, gain, -slot[0], slot[1] == "k", slot, next_idx))
        if not options:
            raise ValueError(f"budget {budget} unreachable for targets {targets}")
        _, gain, _, _, slot, next_idx = max(options)
        indices[slot] = next_idx
        total += gain
    assignment = [choices[slot][indices[slot]] for slot in slots if indices[slot] >= 0]
    return {"budget": budget, "compression": total / len(slots),
            "whole_kv_compression": total / (2 * layers), "assignment": assignment,
            "kv": {"pipeline": [r["kv"]["pipeline"][0] for r in assignment]}}


def task_plan(calibration, layers, out):
    configs = [configuration("dense_ceval", {"pipeline": []}, out / "tasks", CEVAL_VALID_5SHOT)]
    selections = []
    for mode, targets in (("keys", ("k",)), ("kv", ("k", "v"))):
        for budget in GLOBAL_BUDGETS:
            selection = allocate(calibration, layers, targets, budget)
            selection.update(mode=mode, tag=f"{mode}_b{round(budget * 100):02d}")
            selections.append(selection)
            configs.append(configuration(selection["tag"], selection["kv"], out / "tasks", CEVAL_VALID_5SHOT,
                kv_budget=budget, kv_compression=selection["compression"]))
    return selections, {"model": "hf", "model_args": LLAMA31_8B, "batch_size": 1, "configurations": configs}


def execute(path, out):
    subprocess.run([sys.executable, str(ROOT / "engine/multi_run.py"), "--run", str(path),
                    "--skip-existing", "--write-results", "--results-root", str(out)], cwd=ROOT, check=True)


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
        accuracy = payload["results"]["ceval-valid"]["acc,none"]
        rows.append({"mode": selection["mode"], "budget": selection["budget"],
            "planned_compression": selection["compression"],
            "measured_compression": per_slot_saving * selected_slots / denominator,
            "whole_kv_compression": per_slot_saving * selected_slots / (2 * layers),
            "accuracy": accuracy, "delta_accuracy": accuracy - dense, "cosine": measured})
    return {"dense_accuracy": dense, "task": "ceval-valid", "shots": 5, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "compression_topics/spatial/figures/pair_calibrated_sampled")
    parser.add_argument("--local-budgets", type=float, nargs="+", default=list(LOCAL_BUDGETS))
    parser.add_argument("--residual-entries", type=int, nargs="+", default=list(RESIDUAL_ENTRIES))
    parser.add_argument("--screen-chunks", type=int, default=16)
    parser.add_argument("--refine-chunks", type=int, default=32)
    parser.add_argument("--extra-chunks", type=int, default=32, help="additional chunks for unstable or close finalist comparisons; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--from", dest="start", choices=STAGES, default="sweep")
    parser.add_argument("--through", choices=STAGES, default="summary")
    parser.add_argument("--execute", action="store_true", help="explicitly start evaluations; otherwise only write the plan")
    args = parser.parse_args()
    if STAGES.index(args.start) > STAGES.index(args.through):
        parser.error("--from must not follow --through")
    budgets = sorted(set(args.local_budgets))
    if not budgets or any(not 0 < b <= .5 for b in budgets) or any(abs(b * 100 - round(b * 100)) > 1e-9 for b in budgets):
        parser.error("local budgets must be whole percentages in (0, 50%]")
    if args.screen_chunks < 1 or args.refine_chunks < 2 or args.extra_chunks < 0 or args.finalists < 1 or args.seq_len < 2:
        parser.error("need positive screening/finalist counts, at least two refinement chunks, nonnegative extra chunks and seq-len >= 2")
    residuals = sorted(set(args.residual_entries))
    if not residuals or any(not 0 <= r <= 32 for r in residuals):
        parser.error("residual entries must be between 0 and 32")
    manifest = plan(args.out.resolve(), local_budgets=budgets, residuals=residuals,
                    screen_chunks=args.screen_chunks, refine_chunks=args.refine_chunks,
                    extra_chunks=args.extra_chunks, seed=args.seed, seq_len=args.seq_len, finalists=args.finalists)
    if any(not any(c["budget"] == b for c in manifest["candidates"]) for b in budgets):
        parser.error("a local budget has no feasible residual candidate")
    args.out = args.out.resolve()
    existing_path = args.out / "manifest.json"
    if existing_path.exists():
        previous = json.loads(existing_path.read_text())
        fields = ("layers", "local_budgets", "residual_entries", "screen_chunks", "refine_chunks",
                  "extra_chunks", "seed", "seq_len", "finalists", "ranking")
        if any(previous.get(key) != manifest.get(key) for key in fields):
            parser.error("existing output directory has a different calibration grid; use a new --out directory")
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "sweep_run.json", manifest["run"])
    print(f"Plan: {len(manifest['candidates'])} candidates on {args.screen_chunks} random chunks; "
          f"up to {args.finalists} finalists per slot/budget on {args.refine_chunks} additional chunks.")
    print(f"Unstable/close comparisons get {args.extra_chunks} extra chunks; final C-Eval: 8 configurations + dense, full dataset.")
    print(f"Residual entries: {residuals}; local budgets: {budgets}; global budgets: {GLOBAL_BUDGETS}")
    print(f"Output: {args.out}")
    if not args.execute:
        print("Plan only. Add --execute when ready to run.")
        return
    for stage in STAGES[STAGES.index(args.start):STAGES.index(args.through) + 1]:
        if stage == "sweep":
            execute_chunks(args.out / "sweep_run.json")
        elif stage == "shortlist":
            screened = select(manifest, top_k=args.finalists)
            write_json(args.out / "screening.json", screened)
            refined = subset_manifest(manifest, screened["shortlisted"], args.out / "refine",
                                      args.screen_chunks, args.refine_chunks)
            write_json(args.out / "refine_manifest.json", refined)
            write_json(args.out / "refine_run.json", refined["run"])
        elif stage == "refine":
            execute_chunks(args.out / "refine_run.json")
        elif stage == "stabilize":
            refine_manifest = json.loads((args.out / "refine_manifest.json").read_text())
            refined = select(refine_manifest, top_k=args.finalists)
            screened = json.loads((args.out / "screening.json").read_text())
            stability = comparisons(refined, screened)
            write_json(args.out / "refinement.json", {**refined, "stability": stability})
            unstable = {(r["layer"], r["target"], r["budget"]) for r in stability if r["needs_more"]}
            candidates = [r for r in refined["candidates"] if (r["layer"], r["target"], r["budget"]) in unstable]
            if candidates and args.extra_chunks:
                extended = subset_manifest(manifest, candidates, args.out / "stabilize",
                                           args.screen_chunks, args.refine_chunks + args.extra_chunks, reuse_from=True)
                write_json(args.out / "stabilize_manifest.json", extended)
                write_json(args.out / "stabilize_run.json", extended["run"])
                execute_chunks(args.out / "stabilize_run.json")
        elif stage == "select":
            refined = json.loads((args.out / "refinement.json").read_text())
            extended_path = args.out / "stabilize_manifest.json"
            if extended_path.exists():
                extended = select(json.loads(extended_path.read_text()), top_k=args.finalists)
                affected = {(r["layer"], r["target"], r["budget"]) for r in extended["selected"]}
                for key in ("selected", "candidates", "shortlisted"):
                    refined[key] = [r for r in refined[key] if (r["layer"], r["target"], r["budget"]) not in affected] + extended[key]
            screened = json.loads((args.out / "screening.json").read_text())
            refined["stability"] = comparisons(refined, screened, trigger_changed=False)
            # Normalize using each candidate's loss increase over dense on its
            # own chunks. This puts 32- and 64-chunk candidates on one baseline.
            baseline = refined["dense_ppl"]
            for key in ("selected", "candidates", "shortlisted"):
                for row in refined[key]:
                    row.setdefault("measured_ppl", row["ppl"])
                    row["ppl"] = baseline * math.exp(row["delta_nll"])
            write_json(args.out / "calibration.json", refined)
        elif stage == "greedy":
            calibration = json.loads((args.out / "calibration.json").read_text())
            selections, run = task_plan(calibration, manifest["layers"], args.out)
            write_json(args.out / "selections.json", selections)
            write_json(args.out / "tasks_run.json", run)
        elif stage == "tasks":
            execute(args.out / "tasks_run.json", args.out)
        else:
            selections = json.loads((args.out / "selections.json").read_text())
            write_json(args.out / "summary.json", summary(args.out, selections))


if __name__ == "__main__":
    main()
