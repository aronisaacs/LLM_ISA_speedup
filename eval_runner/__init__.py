"""Boilerplate behind multi_run.py: run-list loading, device pick, lm-eval kwargs."""

from eval_runner.device import apply_device, available_device
from eval_runner.execute import evaluate, load_model_if_needed, write_result_json
from eval_runner.run_list import load_run_list, merge, reject_deprecated_kv_keys, split_base_and_runs

__all__ = [
    "apply_device",
    "available_device",
    "evaluate",
    "load_model_if_needed",
    "load_run_list",
    "merge",
    "reject_deprecated_kv_keys",
    "split_base_and_runs",
    "write_result_json",
]
