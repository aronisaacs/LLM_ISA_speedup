#!/usr/bin/env python3
"""Execute a run: a sequence of configurations (model + compression + task).

python multi_run.py --run runs/compression_x_task.py
"""

from __future__ import annotations

import argparse
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
from kv_compress import install, parse_kv_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="Python run (run()) or JSON, e.g. runs/compression_x_task.py",
    )
    args = parser.parse_args()

    base, configurations = split_base_and_configurations(load_run(Path(args.run)))
    lm = None
    loaded_model_key = None

    for configuration in configurations:
        reject_deprecated_kv_keys(base, configuration)
        lm, loaded_model_key, device, model_args = load_model_if_needed(
            lm, loaded_model_key, base, configuration
        )
        kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
        uninstall = install(lm, kv_spec)
        try:
            results = evaluate(lm, base, configuration, kv_spec, device, model_args)
        finally:
            uninstall()
        write_result_json(lm, base, configuration, results)


if __name__ == "__main__":
    main()
