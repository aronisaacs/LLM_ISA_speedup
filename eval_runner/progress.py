"""Newline status for multi_run.py. Per-request counters stay with lm-eval tqdm."""

from __future__ import annotations

import sys

_PRIMARY_METRICS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "pass_at_1,none",
    "word_perplexity,none",
)


def say(message: str) -> None:
    """Print a whole-line event without breaking an active tqdm bar."""
    text = str(message)
    try:
        from tqdm.auto import tqdm

        tqdm.write(text, file=sys.stderr)
    except Exception:
        print(text, file=sys.stderr, flush=True)


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def kv_brief(spec) -> str:
    pipeline = spec.to_dict().get("pipeline") or []
    if not pipeline:
        return "dense"
    parts = []
    for step in pipeline:
        method = step.get("method", "?")
        if method == "sparsify_nm":
            parts.append(f"sparsify {step.get('n', '?')}:{step.get('m', '?')}")
        else:
            parts.append(str(method))
    return " + ".join(parts)


def summarize_scores(results) -> str:
    """One short metric snippet from an lm-eval results payload."""
    table = (results or {}).get("results") or {}
    bits = []
    for task, metrics in table.items():
        if not isinstance(metrics, dict):
            continue
        name, value = _pick_metric(metrics)
        if name is None:
            continue
        if isinstance(value, float):
            shown = f"{value:.3f}"
        else:
            shown = str(value)
        bits.append(f"{task} {name}={shown}")
    return "  ".join(bits) if bits else "no score yet"


def _pick_metric(metrics: dict):
    for name in _PRIMARY_METRICS:
        if name in metrics and metrics[name] is not None:
            return name, metrics[name]
    for key, value in metrics.items():
        if key in {"alias"} or "stderr" in key or not isinstance(value, (int, float)):
            continue
        return key, value
    return None, None
