"""Shared historical-reference checks for simplification studies.

All candidate generation remains study-specific. Holdout gates and reference
comparison are shared so that resume, byte matching, and sample provenance
cannot drift between quad-only and fixed-hierarchy studies.
"""

from pathlib import Path
import math
from compression_topics.spatial.scripts import study as base
from engine.eval_runner.files import write_json
from compression_topics.spatial.algorithms.methods import SpatialConfig

REFERENCE = (
    base.ROOT / "compression_topics/spatial/figures/clean_online_ladder_query_hierarchy"
)


def reference_selection(p):
    saved = next(
        s
        for s in base.load(Path(p["reference"]) / "selections.json")
        if s["budget"] == 0.4
    )
    steps = []
    for old in saved["kv"]["pipeline"]:
        aware = bool(old["k_layers"])
        config = SpatialConfig(
            max_group=8,
            prefill_only=True,
            control=old["control"],
            query_aware=aware,
            query_weights=p["query_weights"] if aware else None,
            query_weights_sha256=p["query_weights_sha256"] if aware else None,
        )
        steps += config.pipeline(
            old["saving"], k_layers=old["k_layers"], v_layers=old["v_layers"]
        )["pipeline"]
    return {"pipeline": steps}


def _holdout_payload(p, manifest, name):
    from engine.eval_runner.chunks import identity

    out = Path(p["out"])
    run = base.load(out / manifest)
    expected = base.chunk_run(p, [], "validation")
    if run["sampling"] != expected["sampling"] or run["model_args"] != p["model_args"]:
        raise RuntimeError(
            "Reference holdout model or sampling differs from current plan"
        )
    config = next(c for c in run["configurations"] if c["name"] == name)
    payload = base.load(out / "validation" / f"{name}.json")
    if not base.identities_match(
        payload.get("simulation", {}), identity(config, run["sampling"], run)
    ):
        raise RuntimeError(f"Holdout identity differs: {name}")
    if payload.get("storage", {}).get("scope") != "simulated_prefill_cache_storage":
        raise RuntimeError("Holdout requires prefill storage accounting")
    return payload


def quality_gate(p):
    out = Path(p["out"])
    ref = _holdout_payload(p, "reference_validation_run.json", "reference")
    fixed = _holdout_payload(p, "validation_run.json", f"{p['arms'][0]}_b40")
    if [(r["chunk"], r["tokens"]) for r in ref["chunk_scores"]] != [
        (r["chunk"], r["tokens"]) for r in fixed["chunk_scores"]
    ] or ref.get("scoring") != fixed.get("scoring"):
        raise RuntimeError("Reference and candidate holdout samples or scoring differ")
    a, b = base.perplexity(ref), base.perplexity(fixed)
    saving_ref = ref["storage"]["targets"]["kv"]["compression"]
    saving_fixed = fixed["storage"]["targets"]["kv"]["compression"]
    passed = (
        all(math.isfinite(v) for v in (a, b, saving_ref, saving_fixed))
        and a > 0
        and b > 0
        and abs(saving_ref - saving_fixed) <= p["matching_tolerance"]
        and b <= a * p["holdout_ppl_ratio_limit"]
    )
    gate = dict(
        passed=passed,
        reference_ppl=a,
        fixed_ppl=b,
        ppl_ratio=b / a,
        reference_saving=saving_ref,
        fixed_saving=saving_fixed,
        definition="heuristic 2% PPL screen and 1-point byte match; not an accuracy guarantee",
    )
    write_json(out / "quality_gate.json", gate)
    print(gate, flush=True)
    if not passed:
        raise RuntimeError(
            "Holdout quality/byte gate failed; full C-Eval not started. Inspect quality_gate.json."
        )


def summarize(out, run, *, require_complete=False, arm):
    report = base.summarize(
        out,
        run,
        require_complete=require_complete,
        scope="fixed residuals by group size per layer K/V; query-aware; prefill-only",
        comparison="historical adaptive reference plus reevaluated common holdout; 2 screening versus reference 4, both 12 refinement",
    )
    old = base.load(REFERENCE / "summary.json")
    ref = next(r for r in old["rows"] if r.get("global_budget") == 0.4)
    report["historical_reference"] = ref
    for r in report["rows"]:
        if r.get("arm") == arm:
            report["reference_comparison"] = dict(
                accuracy_difference_points=100 * (r["accuracy"] - ref["accuracy"]),
                saving_difference_points=100
                * (r["actual_kv_saving"] - ref["actual_prefill_kv_saving"]),
                within_prespecified_accuracy_tolerance=r["accuracy"]
                >= ref["accuracy"] - 0.01,
                matched_within_one_point=abs(
                    r["actual_kv_saving"] - ref["actual_prefill_kv_saving"]
                )
                <= 0.01,
                meets_simplification_target=r["accuracy"] >= ref["accuracy"] - 0.01
                and abs(r["actual_kv_saving"] - ref["actual_prefill_kv_saving"])
                <= 0.01,
                note="interpret accuracy only alongside actual compression; historical reference, not statistical equivalence",
            )
    write_json(
        Path(out)
        / ("summary.json" if report["status"] == "complete" else "preliminary.json"),
        report,
    )
    return report


def run_comparison_stage(
    p,
    stage,
    *,
    candidates,
    allocations,
    tasks,
    summarizer,
    reference_builder=reference_selection,
):
    if stage == "validate":
        config = base.configuration(
            "reference",
            reference_builder(p),
            Path(p["out"]) / "validation",
            base.WIKITEXT_FULL,
        )
        path = Path(p["out"]) / "reference_validation_run.json"
        write_json(path, base.chunk_run(p, [config], "validation"))
        base.execute_chunks(path)
    result = base.run_stage(
        p,
        stage,
        candidates=candidates,
        arms=p["arms"],
        allocations=allocations,
        tasks=tasks,
        summarizer=summarizer,
    )
    if stage == "validate":
        quality_gate(p)
    return result
