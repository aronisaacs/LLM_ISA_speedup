#!/usr/bin/env python3
"""Eval runner: load a run list, attach KV compression, call lm-eval, write results.

python multi_run.py --config run_lists/mac_eval_runs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_runner import (
    evaluate,
    load_model_if_needed,
    load_run_list,
    merge,
    reject_deprecated_kv_keys,
    split_base_and_runs,
    write_result_json,
)
from kv_compress import install, parse_kv_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Python run list (config()) or JSON, e.g. run_lists/mac_eval_runs.py",
    )
    args = parser.parse_args()

    base, runs = split_base_and_runs(load_run_list(Path(args.config)))
    lm = None
    loaded_model_key = None

    for run in runs:
        reject_deprecated_kv_keys(base, run)
        lm, loaded_model_key, device, model_args = load_model_if_needed(
            lm, loaded_model_key, base, run
        )
        kv_spec = parse_kv_spec(merge(base, run, "kv", None))
        uninstall = install(lm, kv_spec)
        try:
            results = evaluate(lm, base, run, kv_spec, device, model_args)
        finally:
            uninstall()
        write_result_json(lm, base, run, results)


if __name__ == "__main__":
    main()
