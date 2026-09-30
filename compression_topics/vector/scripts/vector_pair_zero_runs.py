#!/usr/bin/env python3
"""Mean zero runs on the RoPE-pair mask under blanket key pruning.

One Llama 3.1 8B load, then six runs at 10–60% pair prune on every key layer.
Values are left dense: they are not paired, so they have no half-length mask.
A zero run is one streak of dropped pairs along that mask (length head_dim // 2).
lm-eval applies ``limit`` per CEval subject, so ``limit`` 1 is one question from
each subject. Scores are not written to results.json.

The log prints the measured mean next to the random-placement mean. A match
means the dropped pairs are scattered. A lower measured mean means they clump.

  python compression_topics/vector/scripts/vector_pair_zero_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress_pair  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT  # noqa: E402
from compression_topics.vector.algorithms.vector_compress_pair import (  # noqa: E402
    disable_pair_mask_profile,
    enable_pair_mask_profile,
    random_pair_mask_runs,
    take_pair_mask_profile,
)
from engine.eval_runner import evaluate, load_model_if_needed, merge, reject_deprecated_kv_keys  # noqa: E402
from engine.kv_compress import install, parse_kv_spec  # noqa: E402

PERCENTAGES = (10, 20, 30, 40, 50, 60)
OUT = Path(__file__).resolve().parents[1] / "figures" / "pair_zero_runs.json"


def main() -> None:
    base = {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
    }
    configurations = _configurations()
    enable_pair_mask_profile()
    lm = None
    loaded_model_key = None
    rows = []
    try:
        for configuration in configurations:
            reject_deprecated_kv_keys(base, configuration)
            pct = configuration["kv"]["pipeline"][0]["prune_pct"]
            name = configuration["name"]
            kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
            print(f"run  {name}", flush=True)
            lm, loaded_model_key, device, model_args = load_model_if_needed(
                lm, loaded_model_key, base, configuration
            )
            uninstall = install(lm, kv_spec)
            try:
                evaluate(lm, base, configuration, kv_spec, device, model_args)
            finally:
                uninstall()
            stats = take_pair_mask_profile()
            mask_length = stats["mask_length"]
            random_mean = None if mask_length is None else random_pair_mask_runs(mask_length, pct)
            row = {
                "prune_pct": pct,
                "mean_zero_runs": stats["mean"],
                "random_zero_runs": random_mean,
                "mask_length": mask_length,
                "vectors": stats["vectors"],
                "runs": stats["runs"],
            }
            rows.append(row)
            _write(rows)
            _print_row(row)
    finally:
        disable_pair_mask_profile()
    print("pair-mask zero runs", flush=True)
    for row in rows:
        _print_row(row)
    print(f"wrote  {OUT}", flush=True)


def _configurations() -> list[dict]:
    extra = {key: value for key, value in CEVAL_VALID_5SHOT.items() if key not in {"name_task", "file"}}
    extra["limit"] = 1
    configurations = []
    for pct in PERCENTAGES:
        configuration = {
            "name": f"llama31_ceval_pair_zero_runs_p{pct:02d}",
            "kv": vector_compress_pair(k_layers="all", v_layers=[], prune_pct=pct),
        }
        configuration.update(extra)
        configurations.append(configuration)
    return configurations


def _print_row(row: dict) -> None:
    measured = row["mean_zero_runs"]
    random_mean = row["random_zero_runs"]
    measured_text = "none" if measured is None else f"{measured:.6f}"
    random_text = "none" if random_mean is None else f"{random_mean:.6f}"
    print(
        f"{row['prune_pct']}  measured {measured_text}  random {random_text}  "
        f"mask {row['mask_length']}",
        flush=True,
    )


def _write(rows: list[dict]) -> None:
    payload = {"task": "ceval-valid", "limit": 1, "target": "k", "levels": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
