"""Boilerplate behind multi_run.py: load a run, pick device, call lm-eval."""

from eval_runner.device import apply_device, available_device
from eval_runner.execute import evaluate, load_model_if_needed, write_result_json
from eval_runner.load_run import load_run, merge, reject_deprecated_kv_keys, split_base_and_configurations

__all__ = [
    "apply_device",
    "available_device",
    "evaluate",
    "load_model_if_needed",
    "load_run",
    "merge",
    "reject_deprecated_kv_keys",
    "split_base_and_configurations",
    "write_result_json",
]
