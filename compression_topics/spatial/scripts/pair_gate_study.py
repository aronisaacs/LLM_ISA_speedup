#!/usr/bin/env python3
"""GSM8K accuracy and stored bytes for gated pair merging with an exact residual.

A pair of neighboring tokens is merged when, after the ``keep`` percent of features with the
largest difference are kept exactly, the difference left over is at most ``tau`` times the
pair's mean in length. Otherwise the pair stays as it is. Every layer is gated the same way
(keys only, values only, or both). ``keep`` 0 is the same gate with no residual.

Each run reports the pairs it merged, from which the stored bytes come. The summary lists
accuracy against bytes of the whole KV cache next to the dense run.

  python compression_topics/spatial/scripts/pair_gate_study.py                 # run, then summarize
  python compression_topics/spatial/scripts/pair_gate_study.py --model 1b --limit 40 --results-root /tmp/gate
  python compression_topics/spatial/scripts/pair_gate_study.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import DENSE, pair_gate  # noqa: E402
from catalog.models import BATCH_SIZE, LLAMA31_8B, LLAMA32_1B  # noqa: E402
from catalog.tasks import GSM8K_20PCT  # noqa: E402
from compression_topics.spatial.algorithms.pair_gate import bytes_vs_dense  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "pair_gate"
MODELS = {"8b": ("llama31_8b", LLAMA31_8B), "1b": ("llama32_1b", LLAMA32_1B)}
TARGETS = ("keys", "values", "both")
TAUS = (0.15, 0.3, 0.5)
KEEPS = (0, 25, 50)
METRIC = "exact_match,flexible-extract"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="8b")
    parser.add_argument("--limit", type=int, default=None, help="GSM8K questions per run (default: the full task)")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--taus", nargs="+", type=float, default=list(TAUS))
    parser.add_argument("--keeps", nargs="+", type=int, default=list(KEEPS), help="percent of features kept exactly")
    parser.add_argument("--results-root", type=Path, default=None, help="write results.json here instead of the repo root")
    parser.add_argument("--out", type=Path, default=None, help="where run results go (default: figures/pair_gate/<model>)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    prefix, model_args = MODELS[args.model]
    out = args.out or FIGURES / args.model
    if not args.summary_only:
        _run(_configurations(prefix, args, out), model_args, args)
    print(summary(out, args.model))


def _kv(target: str, tau: float, keep: int) -> dict:
    layers = {"keys": ("all", []), "values": ([], "all"), "both": ("all", "all")}[target]
    return pair_gate(k_layers=layers[0], v_layers=layers[1], tau=tau, keep_pct=keep)


def _name(target: str, tau: float, keep: int) -> str:
    return f"gate_{target}_t{round(tau * 100):03d}_k{keep:03d}"


def _configurations(prefix: str, args, out: Path) -> list[dict]:
    extra = {key: value for key, value in GSM8K_20PCT.items() if key not in {"name_task", "file"}}
    if args.limit is not None:
        extra["limit"] = args.limit
    configurations = [{"name": f"{prefix}_dense", "kv": DENSE, "output_path": str(out / "dense.json"), **extra}]
    for target in args.targets:
        for tau in args.taus:
            for keep in args.keeps:
                name = _name(target, tau, keep)
                configurations.append(
                    {"name": f"{prefix}_{name}", "kv": _kv(target, tau, keep), "output_path": str(out / f"{name}.json"), **extra}
                )
    return configurations


def _run(configurations: list[dict], model_args: str, args) -> None:
    run = {"model": "hf", "batch_size": BATCH_SIZE, "model_args": model_args, "configurations": configurations}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    command = [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"]
    if args.results_root is not None:
        command += ["--results-root", str(args.results_root)]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        path.unlink(missing_ok=True)


def cache_bytes(target: str, stats: dict) -> float:
    """Whole KV cache bytes against dense, assuming K and V are the same size in every layer."""
    keys = bytes_vs_dense(stats["k"]) if "k" in stats else 1.0
    values = bytes_vs_dense(stats["v"]) if "v" in stats else 1.0
    return (keys + values) / 2


def summary(out: Path, model: str) -> str:
    rows = []
    for path in sorted(out.glob("*.json")):
        payload = json.loads(path.read_text())
        score = (payload.get("results") or {}).get("gsm8k", {}).get(METRIC)
        if score is None:
            continue
        stats = payload.get("gate_stats") or {}
        if path.stem == "dense":
            rows.append(("dense", "-", "-", "-", 1.0, score, None))
            continue
        _, target, tau, keep = path.stem.split("_")
        merged = {t: (s["merged"] / s["pairs"] if s["pairs"] else 0.0) for t, s in stats.items()}
        rows.append((target, int(tau[1:]) / 100, int(keep[1:]), merged, cache_bytes(target, stats), score, stats))
    if not rows:
        return f"no finished runs in {out}"
    dense = next((row[5] for row in rows if row[0] == "dense"), None)
    lines = [
        f"GSM8K ({METRIC}), Llama {model}; gated pair merge, every layer",
        "",
        "| target | tau | kept % | pairs merged (k / v) | KV bytes vs dense | accuracy | vs dense |",
        "|---|---|---|---|---|---|---|",
    ]
    for target, tau, keep, merged, size, score, _ in sorted(rows, key=lambda r: (str(r[0]), str(r[1]), str(r[2]))):
        share = "-" if merged == "-" else " / ".join(f"{merged.get(t, 0):.2f}" if t in merged else "-" for t in ("k", "v"))
        change = "" if dense is None or target == "dense" else f"{score - dense:+.3f}"
        lines.append(f"| {target} | {tau} | {keep} | {share} | {size:.3f} | {score:.3f} | {change} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
