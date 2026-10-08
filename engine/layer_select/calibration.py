"""Reusable staged calibration on matched chunks and measured byte budgets.

Study modules supply candidates, comparison policy, final tasks and summaries.
"""
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from engine.eval_runner.files import write_json
from engine.layer_select.scores import word_perplexity

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("sweep", "shortlist", "refine", "stabilize", "select", "greedy", "tasks", "summary")

def configuration(name, kv, out, task, **metadata):
    return {"name": name, "kv": kv, "output_path": str(out / f"{name}.json"),
            "metadata": metadata,
            **{k: v for k, v in task.items() if k not in {"name_task", "file"}}}


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


def subset_manifest(original, candidates, out, offset, chunks, reuse_from=False):
    by_name = {r["name"]: r for r in original["candidates"]}
    selected = [{**by_name[r["name"]], "output_path": str(out / f"{r['name']}.json")} for r in candidates]
    templates = {row["name"]: row for row in original["run"]["configurations"]}
    dense = original["run"]["configurations"][0]
    configs = [{**dense, "output_path": str(out / f"{dense['name']}.json")}]
    configs.extend({**templates[r["name"]], "output_path": r["output_path"]} for r in selected)
    if reuse_from:
        for config, candidate in zip(configs[1:], candidates):
            config["reuse_path"] = candidate["output_path"]
        # Reuse dense on the same 32 refinement chunks as well.
        configs[0]["reuse_path"] = str(Path(candidates[0]["output_path"]).parent / "dense_wikitext.json")
    sampling = {**original["run"]["sampling"], "offset": offset, "chunks": chunks}
    return {**original, "candidates": selected,
            "run": {**original["run"], "configurations": configs, "sampling": sampling}}


def execute_chunks(path):
    subprocess.run([sys.executable, str(ROOT / "engine/eval_runner/chunks.py"),
                    "--run", str(path)], cwd=ROOT, check=True)


def select(manifest, tolerance=.001, top_k=1, group_fields=("layer", "target", "budget"),
           *, measure=measurement, score=perplexity, tie_break=None):
    dense_path = Path(manifest["run"]["configurations"][0]["output_path"])
    dense_ppl = score(json.loads(dense_path.read_text()))
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
        ppl = score(payload)
        measured.append({**candidate, "ppl": ppl, "delta_ppl": ppl - dense_ppl,
                         "delta_nll": math.log(ppl) - math.log(dense_ppl),
                         **measure(payload), "chunk_scores": payload.get("chunk_scores", [])})
    groups = sorted({tuple(r[key] for key in group_fields) for r in manifest["candidates"]})
    for group in groups:
        layer, target, budget = group[:3]
        choices = [r for r in measured if tuple(r[key] for key in group_fields) == group and r["compression"] >= budget - tolerance]
        if not choices:
            raise ValueError(f"no candidate meets L{layer} {target} budget {budget}")
        tie_break = tie_break or (lambda row: (row.get("residual_entries", 0), row["name"]))
        ranked = sorted(choices, key=lambda r: (r["ppl"], tie_break(r)))
        winners.append(ranked[0])
        shortlisted.extend(ranked[:top_k])
    return {"dense_ppl": dense_ppl, "budget_tolerance": tolerance,
            "candidates": measured, "selected": winners, "shortlisted": shortlisted}


def comparisons(calibration, screening, trigger_changed=True, group_fields=("layer", "target", "budget")):
    screened = {tuple(r[key] for key in group_fields): r["name"] for r in screening["selected"]}
    reports = []
    for winner in calibration["selected"]:
        group = tuple(winner[key] for key in group_fields)
        choices = sorted([r for r in calibration["candidates"] if
                          tuple(r[key] for key in group_fields) == group and
                          r["compression"] >= r["budget"] - calibration["budget_tolerance"]],
                         key=lambda r: r["ppl"])
        changed = screened[group] != winner["name"]
        report = {**dict(zip(group_fields, group)),
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


def run_stages(args, manifest, *, select, comparisons, task_plan, execute, summary):
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
            run.update(model=manifest["run"]["model"], model_args=manifest["run"]["model_args"])
            write_json(args.out / "selections.json", selections)
            write_json(args.out / "tasks_run.json", run)
        elif stage == "tasks":
            execute(args.out / "tasks_run.json", args.out)
        else:
            selections = json.loads((args.out / "selections.json").read_text())
            write_json(args.out / "summary.json", summary(args.out, selections))
