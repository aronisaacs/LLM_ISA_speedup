"""Boilerplate behind multi_run.py: load a run, pick device, call lm-eval."""

from engine.eval_runner.device import apply_device, available_device
from engine.eval_runner.execute import (
    evaluate,
    is_finished_result,
    load_model_if_needed,
    result_output_path,
    shorten_result,
    write_result_json,
)
from engine.eval_runner.load_run import grid, load_run, merge, reject_deprecated_kv_keys, split_base_and_configurations

__all__ = [
    "apply_device",
    "available_device",
    "evaluate",
    "grid",
    "is_finished_result",
    "load_model_if_needed",
    "result_output_path",
    "load_run",
    "merge",
    "reject_deprecated_kv_keys",
    "shorten_result",
    "split_base_and_configurations",
    "write_result_json",
]
