"""Turn WikiText sweep scores into a mixed-level layer assignment.

Rank-fill (no GPU), one rung at a time:
  python scripts/select_layers.py --scores results/layer_sweep_wikitext \\
      --space keys_only --algo rank_fill --budget 0.5 \\
      --out results/layer_sweep_wikitext/selected.json

Sequential (live WikiText, cannot skip rungs):
  python scripts/select_layers.py --eval-run runs/layer_sweep_wikitext.py \\
      --space keys_only --algo sequential --budget 0.5 \\
      --out results/layer_sweep_wikitext/selected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_runner import (
    evaluate,
    load_model_if_needed,
    load_run,
    merge,
    reject_deprecated_kv_keys,
    split_base_and_configurations,
    write_result_json,
)
from eval_runner.progress import kv_brief, say, summarize_scores
from kv_compress import install, parse_kv_spec
from layer_select.apply import kv_for_assignment, method_template
from layer_select.greedy import get_algo
from layer_select.greedy.rank_fill import rank_fill
from layer_select.greedy.sequential import run_sequential
from layer_select.levels import DEFAULT_LEVELS
from layer_select.scores import kv_from_payload, load_sweep_scores, word_perplexity
from layer_select.slots import Slot, n_hidden_layers, space_slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="rank_fill", help="rank_fill or sequential")
    parser.add_argument("--space", default="keys_only", help="keys_only or mix")
    parser.add_argument(
        "--budget",
        type=float,
        default=0.5,
        help="Mean compression across slots in [0, 1], e.g. 0.5 = 50%% average",
    )
    parser.add_argument("--out", required=True, help="Write selected.json here")
    parser.add_argument("--scores", help="Directory of singleton sweep JSONs (rank_fill)")
    parser.add_argument("--eval-run", help="Run file with model/task defaults (sequential)")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated percents, default 15,30,40,50,60",
    )
    parser.add_argument("--trials-dir", default=None, help="Optional folder for sequential trial JSONs")
    args = parser.parse_args()
    get_algo(args.algo)
    levels = _parse_levels(args.levels)

    if args.algo == "rank_fill":
        payload = _rank_fill(args, levels)
    elif args.algo == "sequential":
        payload = _sequential(args, levels)
    else:
        raise ValueError(f"unsupported algo {args.algo!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    say(f"wrote  {out}  assignment={payload['assignment']}")


def _parse_levels(text) -> tuple[int, ...]:
    if not text:
        return DEFAULT_LEVELS
    return tuple(int(part.strip()) for part in str(text).split(",") if part.strip())


def _rank_fill(args, levels) -> dict:
    if not args.scores:
        raise SystemExit("rank_fill requires --scores")
    dense_ppl, rows = load_sweep_scores(args.scores)
    n_layers = args.n_layers or _n_layers_from_rows(rows)
    slots = space_slots(args.space, n_layers)
    assignment = rank_fill(rows, slots, args.budget, levels=levels, dense_ppl=dense_ppl)
    template = method_template(kv_from_payload(json.loads(Path(rows[0].path).read_text())))
    return _selected_payload(
        algo="rank_fill",
        space=args.space,
        budget=args.budget,
        n_layers=n_layers,
        levels=levels,
        assignment=assignment,
        method_kv=template,
        extra={"dense_ppl": dense_ppl},
    )


def _sequential(args, levels) -> dict:
    if not args.eval_run:
        raise SystemExit("sequential requires --eval-run")
    run_path = Path(args.eval_run)
    base, configurations = split_base_and_configurations(load_run(run_path))
    template = _template_from_configurations(configurations)
    n_layers = args.n_layers or n_hidden_layers(merge(base, {}, "model_args", ""))
    slots = space_slots(args.space, n_layers)
    trials_dir = Path(args.trials_dir) if args.trials_dir else None
    trial_extra = _eval_extra_from_configurations(configurations)

    lm = None
    loaded_model_key = None

    def evaluate_assignment(assignment: dict[Slot, int]) -> float:
        nonlocal lm, loaded_model_key
        tag = "dense" if not assignment else "_".join(
            f"{slot.tag()}p{pct}" for slot, pct in sorted(assignment.items())
        )
        configuration = {
            "name": f"sequential_{tag}",
            "kv": kv_for_assignment(template, assignment),
        }
        configuration.update(trial_extra)
        if trials_dir is not None:
            configuration["output_path"] = str(trials_dir / f"{configuration['name']}.json")
        reject_deprecated_kv_keys(base, configuration)
        lm, loaded_model_key, device, model_args = load_model_if_needed(
            lm, loaded_model_key, base, configuration
        )
        kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
        say(f"trial  {configuration['name']}  {kv_brief(kv_spec)}")
        uninstall = install(lm, kv_spec)
        try:
            results = evaluate(lm, base, configuration, kv_spec, device, model_args)
        finally:
            uninstall()
        if trials_dir is not None:
            write_result_json(lm, base, configuration, results)
        say(f"trial  {summarize_scores(results)}")
        return word_perplexity(results)

    assignment = run_sequential(
        slots=slots,
        budget=args.budget,
        evaluate_assignment=evaluate_assignment,
        levels=levels,
    )
    return _selected_payload(
        algo="sequential",
        space=args.space,
        budget=args.budget,
        n_layers=n_layers,
        levels=levels,
        assignment=assignment,
        method_kv=template,
        extra={"eval_run": str(run_path)},
    )


def _eval_extra_from_configurations(configurations: list) -> dict:
    skip = {"name", "kv", "output_path", "layer_slot", "compression_level"}
    for configuration in configurations:
        extra = {key: value for key, value in configuration.items() if key not in skip}
        if extra.get("tasks"):
            return extra
    return {}


def _template_from_configurations(configurations: list) -> dict:
    for configuration in configurations:
        kv = configuration.get("kv") or {}
        if kv.get("pipeline"):
            return method_template(kv)
    raise ValueError("eval run has no non-dense method to apply")


def _n_layers_from_rows(rows) -> int:
    if not rows:
        raise ValueError("no singleton scores")
    return max(row.slot.layer for row in rows) + 1


def _selected_payload(*, algo, space, budget, n_layers, levels, assignment, method_kv, extra):
    kv = kv_for_assignment(method_kv, assignment)
    payload = {
        "algo": algo,
        "space": space,
        "budget": budget,
        "n_layers": n_layers,
        "levels": list(levels),
        "assignment": [
            {**slot.to_dict(), "level": pct} for slot, pct in sorted(assignment.items())
        ],
        "kv": kv,
    }
    payload.update(extra)
    return payload


if __name__ == "__main__":
    main()
