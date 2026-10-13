"""Shared staged protocol for spatial accuracy studies.

Stages: screen -> refine -> allocate -> validate -> tasks -> summary.
Each arm calibrates single layer slots (one K or V slot compressed at a time) on
WikiText train chunks: 4 screening chunks, then the screen winners on 12 disjoint
refinement chunks. A common greedy allocator turns each arm's measured
perplexity/byte curves into whole-model policies. A holdout of validation chunks
must reproduce every global saving within one point before full five-shot C-Eval
runs. Dense C-Eval is reused only after an exact identity and question-count check.
Plans are immutable per output directory; use a fresh ``--out`` for a new protocol.
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from catalog.models import LLAMA31_8B
from catalog.tasks import CEVAL_VALID_5SHOT, WIKITEXT_FULL
from compression_topics.spatial.algorithms.controller import Control
from compression_topics.spatial.algorithms.formats import ACCOUNTING, MENUS
from engine.eval_runner.cache import identities_match, simulation_identity
from engine.eval_runner.files import write_json
from engine.eval_runner.index import record_simulation
from engine.kv_compress.spec import parse_kv_spec
from engine.layer_select.calibration import configuration, execute_chunks, perplexity
from engine.layer_select.study import (
    allocate_settings,
    measure_candidates,
    require_matched_savings,
    select_settings,
)

STAGES = ("screen", "refine", "allocate", "validate", "tasks", "summary")
DENSE_CEVAL = (
    ROOT
    / "compression_topics/spatial/figures/online_pairs_accuracy_offline/tasks/dense_ceval.json"
)


def base_plan(
    out, *, layers=32, budgets=(0.3, 0.4, 0.5), screen=4, refine=12, validation=16
):
    return dict(
        out=str(out),
        layers=layers,
        head_dim=128,
        model_args=LLAMA31_8B,
        budgets=list(budgets),
        screen_chunks=screen,
        refine_chunks=refine,
        validation_chunks=validation,
        seed=83,
        seq_len=2048,
        scoring_prefix=1536,
        control=asdict(Control()),
        menus={str(g): list(m) for g, m in MENUS.items()},
        accounting=ACCOUNTING,
        allocation_basis="measured_calibration_bytes",
        matching_tolerance=0.01,
        dense_ceval_source=str(DENSE_CEVAL),
    )


def load(path):
    return json.loads(Path(path).read_text())


def write_plan(study):
    path = Path(study["out"]) / "plan.json"
    if path.exists() and load(path) != study:
        raise SystemExit(
            "a different protocol exists in this output directory; choose a fresh --out"
        )
    write_json(path, study)


def slot(layer, target):
    return dict(
        k_layers=[layer] if target == "k" else [],
        v_layers=[layer] if target == "v" else [],
    )


def chunk_run(study, configs, stage):
    count = study[stage + "_chunks"]
    sample = {
        "split": "validation" if stage == "validation" else "train",
        "pool_chunks": (
            count
            if stage == "validation"
            else study["screen_chunks"] + study["refine_chunks"]
        ),
        "offset": study["screen_chunks"] if stage == "refine" else 0,
        "chunks": count,
        "seed": study["seed"],
        "seq_len": study["seq_len"],
        "scoring_prefix": study["scoring_prefix"],
    }
    return {
        "model": "hf",
        "model_args": study["model_args"],
        "batch_size": 1,
        "model_shape": {"layers": study["layers"], "head_dim": study["head_dim"]},
        "sampling": sample,
        "preliminary_results": True,
        "configurations": configs,
    }


def slot_measurement(payload):
    """Measured bytes of the single compressed slot of a calibration run."""
    stats = {
        key: value
        for key, value in payload.get("gate_stats", {}).items()
        if "_layer_" in key
    }
    if len(stats) != 1:
        raise ValueError("calibration must measure exactly one KV slot")
    row = next(iter(stats.values()))
    if row.get("accounting") != ACCOUNTING or row.get("dense_bits", 0) <= 0:
        raise ValueError("missing spatial byte accounting")
    from engine.eval_runner.storage import calibration_slot_bytes

    key = next(iter(stats))
    counts, source = calibration_slot_bytes(payload, key)
    return dict(
        compression=counts["compression"],
        shortfall_updates=0,
        stats=stats,
        measured_dense_bits=counts["dense_bits"],
        measured_stored_bits=counts["stored_bits"],
        measurement_scope="prefill" if payload.get("scoring") else "cache",
        measurement_source=source,
    )


def measure(study, rows, stage):
    return measure_candidates(
        rows,
        Path(study["out"]) / stage,
        tolerance=0.03,
        budget_rule="absolute",
        measurement_fn=slot_measurement,
        budget_failure="exclude",
    )


def winners(calibration, arm):
    return select_settings(calibration, include=lambda row: row["arm"] == arm)


def allocate(study, calibration, arm, budgets=None):
    """Measured-byte greedy allocation for one arm at each global budget."""
    settings = winners(calibration, arm)
    best = {}
    for row in settings:  # equal measured bytes: keep the better perplexity
        key = (row["layer"], row["target"], row["measured_saving"])
        if key not in best or row["ppl"] < best[key]["ppl"]:
            best[key] = row
    result = []
    for budget in budgets or study["budgets"]:
        selection = allocate_settings(
            calibration,
            list(best.values()),
            study["layers"],
            ("k", "v"),
            budget,
            allocation_basis="measured",
        )
        selection.update(arm=arm, tag=f"{arm}_b{round(100 * budget):02d}")
        result.append(selection)
    return result


def task_run(study, selections):
    out = Path(study["out"])
    configs = [
        configuration("dense_ceval", {"pipeline": []}, out / "tasks", CEVAL_VALID_5SHOT)
    ]
    configs += [
        configuration(
            s["tag"] + "_ceval",
            s["kv"],
            out / "tasks",
            CEVAL_VALID_5SHOT,
            arm=s["arm"],
            global_budget=s["budget"],
        )
        for s in selections
    ]
    run = dict(
        model="hf", model_args=study["model_args"], batch_size=1, configurations=configs
    )
    source = Path(study["dense_ceval_source"])
    if source.exists():
        payload = load(source)
        wanted = simulation_identity(run, configs[0], parse_kv_spec(configs[0]["kv"]))
        if (
            identities_match(payload.get("simulation", {}), wanted)
            and payload.get("n-samples", {}).get("ceval-valid", {}).get("effective")
            == 1346
        ):
            write_json(configs[0]["output_path"], payload)
            record_simulation(
                wanted, payload["results"], payload["n-samples"], root=out
            )
            print("Reusing identity-checked full dense C-Eval baseline.", flush=True)
    return run


def summarize(
    out, run, *, require_complete=False, arm_pairs=(), scope="", comparison=""
):
    out = Path(out)
    rows = []
    for config in run["configurations"]:
        path = Path(config["output_path"])
        if not path.exists():
            continue
        payload = load(path)
        wanted = simulation_identity(run, config, parse_kv_spec(config["kv"]))
        if not identities_match(payload.get("simulation", {}), wanted):
            raise ValueError(f"result identity differs: {path}")
        count = payload["n-samples"]["ceval-valid"]["effective"]
        if count != 1346:
            raise ValueError(f"expected full 1346-question C-Eval, found {count}")
        stats = [
            r
            for r in payload.get("gate_stats", {}).values()
            if "controller_bound_hits" in r
        ]
        saving = (
            payload["storage"]["targets"]["kv"]["compression"]
            if config["kv"]["pipeline"]
            else 0.0
        )
        row = {
            "name": config["name"],
            **config.get("metadata", {}),
            "accuracy": payload["results"]["ceval-valid"]["acc,none"],
            "questions": count,
            "actual_kv_saving": saving,
            "max_local_target_deviation": max(
                (r["max_final_target_deviation"] for r in stats), default=None
            ),
            "controller_bound_hits": sum(r["controller_bound_hits"] for r in stats),
        }
        if "global_budget" in row:
            row["saving_deviation_points"] = 100 * (saving - row["global_budget"])
        rows.append(row)
    dense = next((r["accuracy"] for r in rows if r["name"] == "dense_ceval"), None)
    for row in rows:
        row["accuracy_change_points"] = (
            100 * (row["accuracy"] - dense) if dense is not None else None
        )
    complete = len(rows) == len(run["configurations"])
    if require_complete and not complete:
        raise ValueError("accuracy run incomplete")
    comparisons = []
    for budget in sorted({r["global_budget"] for r in rows if "global_budget" in r}):
        by_arm = {r["arm"]: r for r in rows if r.get("global_budget") == budget}
        for left, right in arm_pairs:
            if left in by_arm and right in by_arm:
                gap = (
                    by_arm[right]["actual_kv_saving"] - by_arm[left]["actual_kv_saving"]
                )
                comparisons.append(
                    {
                        "budget": budget,
                        "reference_arm": left,
                        "comparison_arm": right,
                        "saving_gap_points": 100 * gap,
                        "matched_within_one_point": abs(gap) <= 0.01,
                        "accuracy_gain_points": 100
                        * (by_arm[right]["accuracy"] - by_arm[left]["accuracy"]),
                    }
                )
    report = {
        "status": "complete" if complete else "preliminary",
        "completed": len(rows),
        "total": len(run["configurations"]),
        "dense_accuracy": dense,
        "rows": rows,
        "comparisons": comparisons,
        "scope": scope,
        "comparison": comparison,
    }
    write_json(out / ("summary.json" if complete else "preliminary.json"), report)
    return report


def execute_tasks(out, run, *, summarizer=summarize):
    command = [
        sys.executable,
        str(ROOT / "engine/multi_run.py"),
        "--run",
        str(Path(out) / "tasks_run.json"),
        "--skip-existing",
        "--write-results",
        "--results-root",
        str(out),
    ]
    process = subprocess.Popen(command, cwd=ROOT)
    previous = -1
    try:
        while True:
            report = summarizer(out, run)
            if report["completed"] != previous:
                print("[preliminary] " + json.dumps(report), flush=True)
                previous = report["completed"]
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                pass
        if code:
            raise subprocess.CalledProcessError(code, command)
        print(
            json.dumps(summarizer(out, run, require_complete=True), indent=2),
            flush=True,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def read_validation(study, selections):
    """Validate raw holdout identities against the policies about to be evaluated."""
    from engine.eval_runner.chunks import identity

    out = Path(study["out"])
    run = load(out / "validation_run.json")
    expected = chunk_run(study, [], "validation")
    if (
        run["sampling"] != expected["sampling"]
        or run["model_args"] != expected["model_args"]
    ):
        raise ValueError("validation model or sampling differs from current plan")
    configs = {c["name"]: c for c in run["configurations"]}
    result = []
    for selection in selections:
        config = configs[selection["tag"]]
        if config["kv"] != selection["kv"]:
            raise ValueError("selected policy differs from validated policy")
        payload = load(out / "validation" / f"{selection['tag']}.json")
        if not identities_match(
            payload.get("simulation", {}), identity(config, run["sampling"], run)
        ):
            raise ValueError("validation result identity differs")
        if (
            study.get("scoring_prefix") is not None
            and payload.get("storage", {}).get("scope")
            != "simulated_prefill_cache_storage"
        ):
            raise ValueError("validation needs prefill storage")
        result.append(
            dict(
                arm=selection["arm"],
                budget=selection["budget"],
                actual_saving=payload["storage"]["targets"]["kv"]["compression"],
                token_ppl=perplexity(payload),
            )
        )
    return result


def run_stage(
    study,
    stage,
    *,
    candidates,
    arms,
    allocations,
    tasks,
    summarizer,
    validation_checker=require_matched_savings,
):
    out = Path(study["out"])
    if stage in ("screen", "refine", "validate"):
        name = "validation" if stage == "validate" else stage
        destination = out / name
        configs = [configuration("dense", {"pipeline": []}, destination, WIKITEXT_FULL)]
        rows = candidates(study)
        if stage == "refine":
            shortlist = set(load(out / "shortlist.json"))
            rows = [r for r in rows if r["name"] in shortlist]
        if stage == "validate":
            selections = load(out / "selections.json")
            configs += [
                configuration(s["tag"], s["kv"], destination, WIKITEXT_FULL)
                for s in selections
            ]
        else:
            configs += [
                configuration(r["name"], r["kv"], destination, WIKITEXT_FULL)
                for r in rows
            ]
        path = out / f"{name}_run.json"
        write_json(path, chunk_run(study, configs, name))
        execute_chunks(path)
        if stage in ("screen", "refine"):
            calibration = measure(study, rows, name)
            write_json(
                out / ("screening.json" if stage == "screen" else "calibration.json"),
                calibration,
            )
            rejected = calibration.get("rejected_candidates", [])
            write_json(out / f"{name}_rejected_candidates.json", rejected)
            print(
                f'{name}: {len(calibration["candidates"])} accepted, {len(rejected)} rejected local byte targets.',
                flush=True,
            )
            if not calibration["candidates"]:
                raise RuntimeError(
                    "No usable local candidates; inspect rejected_candidates before allocation"
                )
            if stage == "screen":
                chosen = [r["name"] for arm in arms for r in winners(calibration, arm)]
                write_json(out / "shortlist.json", chosen)
                print(f"Shortlisted {len(chosen)}/{len(rows)} candidates.", flush=True)
        else:
            validation = read_validation(study, selections)
            write_json(out / "validation.json", validation)
            try:
                validation_checker(
                    validation,
                    study["budgets"],
                    arms,
                    tolerance=study["matching_tolerance"],
                )
            except ValueError as error:
                raise RuntimeError(
                    "holdout byte gate failed; inspect validation.json before full accuracy"
                ) from error
            print(json.dumps(validation, indent=2), flush=True)
    elif stage == "allocate":
        write_json(
            out / "selections.json", allocations(study, load(out / "calibration.json"))
        )
    elif stage == "tasks":
        selections = load(out / "selections.json")
        try:
            validation = read_validation(study, selections)
            if validation != load(out / "validation.json"):
                raise ValueError("validation summary differs from raw results")
        except (ValueError, KeyError) as error:
            raise RuntimeError("validation provenance check failed") from error
        try:
            validation_checker(
                validation,
                study["budgets"],
                arms,
                tolerance=study["matching_tolerance"],
            )
        except ValueError as error:
            raise RuntimeError("validation byte gate failed") from error
        run = tasks(study, selections)
        write_json(out / "tasks_run.json", run)
        execute_tasks(out, run, summarizer=summarizer)
    else:
        summarizer(out, load(out / "tasks_run.json"), require_complete=True)


def run_stages(study, start, through, run):
    if STAGES.index(start) > STAGES.index(through):
        raise SystemExit("invalid stage order")
    for stage in STAGES[STAGES.index(start) : STAGES.index(through) + 1]:
        print(f"=== {stage} ===", flush=True)
        run(study, stage)


def calibrate_queries(study):
    """E[q^2] per layer and KV head from fresh dense train prefixes; saves only the small weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache
    from engine.kv_compress.importance import capture_queries
    from engine.eval_runner.text_chunks import wikitext_chunks

    path = Path(study["query_weights"])
    if path.exists():
        if torch.load(path, weights_only=True).get("protocol") != study:
            raise ValueError("query weight provenance mismatch")
        return
    name, dtype = _model_name_dtype(study["model_args"])
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cuda").eval()
    prompts, chosen = wikitext_chunks(
        tokenizer, study["query_chunks"], study["seq_len"], study["seed"], split="train"
    )
    take, uninstall = capture_queries("llama")
    original = Cache.update
    sums, counts = {}, {}

    def update(cache, keys, values, layer, *args, **kwargs):
        q = take().detach().float()[..., : study["query_prefix"], :]
        grouped = q.reshape(1, keys.shape[1], -1, q.shape[-2], q.shape[-1])
        sums[layer] = sums.get(layer, 0) + grouped.square().sum((0, 2, 3)).cpu()
        counts[layer] = counts.get(layer, 0) + q.shape[-2]
        return original(cache, keys, values, layer, *args, **kwargs)

    Cache.update = update
    try:
        with torch.no_grad():
            for n, prompt in enumerate(prompts):
                model(
                    input_ids=prompt[: study["query_prefix"]].unsqueeze(0).to("cuda"),
                    use_cache=True,
                )
                print(f"Query calibration {n + 1}/{len(prompts)}", flush=True)
    finally:
        Cache.update = original
        uninstall()
    weights = torch.stack([sums[i] / counts[i] for i in range(study["layers"])])
    torch.save(dict(weights=weights, chunks=chosen, protocol=study), path)


def _model_name_dtype(model_args):
    import torch

    fields = dict(part.split("=", 1) for part in model_args.split(","))
    return fields["pretrained"], getattr(torch, fields.get("dtype", "bfloat16"))
