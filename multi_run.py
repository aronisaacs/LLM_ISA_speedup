#!/usr/bin/env python3
"""Execute a run: a sequence of configurations (model + compression + task).

python multi_run.py --run runs/compression_x_task.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from eval_runner import (
    evaluate,
    is_finished_result,
    load_model_if_needed,
    load_run,
    merge,
    reject_deprecated_kv_keys,
    result_output_path,
    split_base_and_configurations,
    write_result_json,
)
from eval_runner.progress import format_hms, kv_brief, say, summarize_scores
from kv_compress import install, parse_kv_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="Python run (run()) or JSON, e.g. runs/compression_x_task.py",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a configuration whose output JSON already has task scores",
    )
    args = parser.parse_args()

    run_path = Path(args.run)
    base, configurations = split_base_and_configurations(load_run(run_path))
    total = len(configurations)
    say(f"run  {run_path.stem}  {total} configuration{'s' if total != 1 else ''}")

    lm = None
    loaded_model_key = None
    started = time.monotonic()
    finished = 0

    for index, configuration in enumerate(configurations, start=1):
        reject_deprecated_kv_keys(base, configuration)
        name = configuration.get("name", "configuration")
        output_path = result_output_path(base, configuration)
        if args.skip_existing and is_finished_result(output_path):
            say(f"[{index}/{total}]  skip  {name}  {output_path}")
            continue
        previous_key = loaded_model_key
        lm, loaded_model_key, device, model_args = load_model_if_needed(
            lm, loaded_model_key, base, configuration
        )
        if loaded_model_key != previous_key:
            say(f"loaded  {device}  {model_args}")

        kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
        say(f"[{index}/{total}]  {name}  {kv_brief(kv_spec)}  {device}")

        uninstall = install(lm, kv_spec)
        try:
            results = evaluate(lm, base, configuration, kv_spec, device, model_args)
        finally:
            uninstall()
        output_path = write_result_json(lm, base, configuration, results)

        finished += 1
        elapsed = time.monotonic() - started
        remaining_configs = total - index
        remaining = (elapsed / finished) * remaining_configs if finished else 0
        wrote = output_path if output_path is not None else "(rank skipped write)"
        say(
            f"done   {wrote}  {summarize_scores(results)}  "
            f"elapsed {format_hms(elapsed)}  eta {format_hms(remaining)}"
        )


if __name__ == "__main__":
    main()
