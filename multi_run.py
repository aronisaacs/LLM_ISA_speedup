#!/usr/bin/env python3
"""Execute a run: a sequence of configurations (model + compression + task).

One process per visible CUDA device. Each process keeps the model loaded and
walks its own slice of the list (index modulo the device count). Set
CUDA_VISIBLE_DEVICES to choose which GPUs take part. One GPU, MPS, or CPU
runs in this process.

python multi_run.py --run runs/compression_x_task.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
    args = _parse_args()
    if args.worker is None:
        devices = visible_cuda_devices()
        if len(devices) > 1:
            _spawn_workers(args, devices)
            return
    _run_configurations(args)


def visible_cuda_devices() -> list[str]:
    """Ids in CUDA_VISIBLE_DEVICES, or ``0..n-1`` when that variable is unset."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(index) for index in range(_cuda_device_count())]


def configurations_for_worker(configurations, worker: int, workers: int):
    """Global 1-based index plus configuration for this worker's fixed slice."""
    _check_worker(worker, workers)
    return [
        (index, configuration)
        for index, configuration in enumerate(configurations, start=1)
        if (index - 1) % workers == worker
    ]


def worker_command(run: str, worker: int, workers: int, skip_existing: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run",
        run,
        "--worker",
        str(worker),
        "--workers",
        str(workers),
    ]
    if skip_existing:
        command.append("--skip-existing")
    return command


def _parse_args():
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
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.worker is None) != (args.workers is None):
        parser.error("--worker and --workers must be passed together")
    if args.worker is not None:
        _check_worker(args.worker, args.workers)
    return args


def _spawn_workers(args, devices: list[str]) -> None:
    say(f"workers  {len(devices)} gpus  {','.join(devices)}")
    processes = []
    for worker, device in enumerate(devices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        command = worker_command(args.run, worker, len(devices), args.skip_existing)
        processes.append((device, subprocess.Popen(command, env=env)))
    failed = []
    for device, process in processes:
        code = process.wait()
        if code != 0:
            failed.append((device, code))
    if failed:
        detail = ", ".join(f"gpu{device}={code}" for device, code in failed)
        say(f"failed  {detail}")
        raise SystemExit(failed[0][1])


def _run_configurations(args) -> None:
    worker = 0 if args.worker is None else args.worker
    workers = 1 if args.workers is None else args.workers
    run_path = Path(args.run)
    base, configurations = split_base_and_configurations(load_run(run_path))
    total = len(configurations)
    mine = configurations_for_worker(configurations, worker, workers)
    label = _device_label(workers)
    if label is None:
        say(f"run  {run_path.stem}  {total} configuration{'s' if total != 1 else ''}")
    else:
        say(f"{label}  {len(mine)} of {total} configurations")

    lm = None
    loaded_model_key = None
    started = time.monotonic()
    finished = 0

    for offset, (index, configuration) in enumerate(mine, start=1):
        reject_deprecated_kv_keys(base, configuration)
        name = configuration.get("name", "configuration")
        output_path = result_output_path(base, configuration)
        if args.skip_existing and is_finished_result(output_path):
            _say(label, f"[{index}/{total}]  skip  {name}  {output_path}")
            continue
        previous_key = loaded_model_key
        lm, loaded_model_key, device, model_args = load_model_if_needed(
            lm, loaded_model_key, base, configuration
        )
        if loaded_model_key != previous_key:
            _say(label, f"loaded  {device}  {model_args}")

        kv_spec = parse_kv_spec(merge(base, configuration, "kv", None))
        _say(label, f"[{index}/{total}]  {name}  {kv_brief(kv_spec)}  {device}")

        uninstall = install(lm, kv_spec)
        try:
            results = evaluate(lm, base, configuration, kv_spec, device, model_args)
        finally:
            uninstall()
        output_path = write_result_json(lm, base, configuration, results)

        finished += 1
        elapsed = time.monotonic() - started
        remaining_configs = len(mine) - offset
        remaining = (elapsed / finished) * remaining_configs if finished else 0
        wrote = output_path if output_path is not None else "(rank skipped write)"
        _say(
            label,
            f"done   {wrote}  {summarize_scores(results)}  "
            f"elapsed {format_hms(elapsed)}  eta {format_hms(remaining)}",
        )


def _device_label(workers: int) -> str | None:
    if workers <= 1:
        return None
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return f"gpu{device}" if device else None


def _say(label: str | None, message: str) -> None:
    if label:
        say(f"{label}  {message}")
    else:
        say(message)


def _check_worker(worker: int, workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    if worker < 0 or worker >= workers:
        raise ValueError(f"worker must be in 0..{workers - 1}")


def _cuda_device_count() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


if __name__ == "__main__":
    main()
