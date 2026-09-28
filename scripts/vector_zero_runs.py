#!/usr/bin/env python3
"""Mean contiguous zero runs under blanket vector compression on CEval.

One Llama 3.1 8B load, then six runs at 10–60% prune on every key and value
layer. lm-eval applies ``limit`` per CEval subject, so ``limit`` 1 is one
question from each subject (about 52), the closest cap to 64 that still covers
every subject. Scores are not written to results.json.

  python scripts/vector_zero_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT  # noqa: E402
from engine.eval_runner import evaluate, load_model_if_needed, merge, reject_deprecated_kv_keys  # noqa: E402
from engine.kv_compress import install, parse_kv_spec  # noqa: E402
from engine.kv_compress.methods.vector_compress import (  # noqa: E402
    disable_zero_run_profile,
    enable_zero_run_profile,
    take_zero_run_profile,
)

PERCENTAGES = (10, 20, 30, 40, 50, 60)
OUT = ROOT / "figures" / "zero_runs.json"


def main() -> None:
    base = {
        "model": "hf",
        "batch_size": BATCH_SIZE,
        "model_args": LLAMA31_8B,
    }
    configurations = _configurations()
    enable_zero_run_profile()
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
            stats = take_zero_run_profile()
            row = {
                "prune_pct": pct,
                "mean_zero_runs": stats["mean"],
                "vectors": stats["vectors"],
                "runs": stats["runs"],
            }
            rows.append(row)
            _write(rows)
            print(f"{pct}  {stats['mean']:.6f}", flush=True)
    finally:
        disable_zero_run_profile()
    print("zero runs", flush=True)
    for row in rows:
        print(f"{row['prune_pct']}  {row['mean_zero_runs']:.6f}", flush=True)
    print(f"wrote  {OUT}", flush=True)


def _configurations() -> list[dict]:
    extra = {key: value for key, value in CEVAL_VALID_5SHOT.items() if key not in {"name_task", "file"}}
    extra["limit"] = 1
    configurations = []
    for pct in PERCENTAGES:
        configuration = {
            "name": f"llama31_ceval_zero_runs_p{pct:02d}",
            "kv": vector_compress(prune_pct=pct),
        }
        configuration.update(extra)
        configurations.append(configuration)
    return configurations


def _write(rows: list[dict]) -> None:
    payload = {"task": "ceval-valid", "limit": 1, "levels": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
