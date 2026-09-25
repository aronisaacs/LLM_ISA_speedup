"""Newline status for multi_run.py. Per-request counters stay with lm-eval tqdm."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_PRIMARY_METRICS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "pass_at_1,none",
    "word_perplexity,none",
)


def mirror_terminal(root: Path) -> Path:
    """Copy stdout and stderr into ``logs/multi_run.log`` as well as the terminal.

    The first process truncates the file. Workers append, so a tmux session
    that disappears still leaves the traceback on disk.
    """
    path = Path(root) / "logs" / "multi_run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = os.environ.get("LLM_ISA_LOG") != "1"
    os.environ["LLM_ISA_LOG"] = "1"
    handle = open(path, "w" if fresh else "a", encoding="utf-8", buffering=1)
    if fresh:
        handle.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {sys.argv}\n")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    return path


class _Tee:
    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        self._handle.flush()

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()


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
        elif method == "checksparse_l1":
            parts.append(f"checksparse L1 tile={step.get('tile', '?')} prune={step.get('prune_pct', '?')}%")
        elif method == "vector_compress":
            parts.append(f"vector |x|<{step.get('threshold', '?')}")
        elif method in {"spatial_pool", "spatial_top1", "spatial_tile", "spatial_feature"}:
            rope = "pre-rope" if step.get("pre_rope") else "post-rope"
            parts.append(f"{method} chunk={step.get('chunk', '?')} {rope}")
        elif method == "qjl":
            parts.append(f"hadamard bits={step.get('bits', '?')}")
        elif method == "dynamic_precision":
            bits = step.get("bits", "?")
            pcts = step.get("pcts", "?")
            parts.append(f"dynprec tile={step.get('tile', '?')} bits={bits} pcts={pcts}")
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
