#!/usr/bin/env python3
"""Scale check (Aron, 2026-11-26): does the length of a rebuilt vector matter?

Spatial stores one factor per token. "length" (the method) makes each rebuilt token exactly as long as
the original, "norm" uses the original norm itself (the rebuilt token comes out slightly shorter, and its
dot products with a query carry no length bias), "lsq" the least-squares factor for its rebuilt direction.
Vector compression stores the kept values as they are ("none", the method; already the least-squares
choice); "length" scales them up so the vector keeps its original length (folded into the values, free).

For each model, the greedy maps of ``rerun.py`` (``results/<model>/<method>/selected_<tag>.json``, their
``calibration_kv``) are scored as they are and with the other factors: KL(dense || compressed) per domain
of the model's 4-domain calibration set, and kl_mean against the method's reference map from
``singles.json`` (same set; lower is better). No task runs. Output: ``results/scale_check/<model>.json``,
resumable, and a table on stdout.

  python compression_topics/kl_greedy/scripts/scale_check.py                     # all three models
  python compression_topics/kl_greedy/scripts/scale_check.py --models llama --timing 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.kl_greedy.scripts.rerun import (  # noqa: E402
    MODELS, OUT, _devices, _write_atomic, calibration_dir)

# Maps around each method's operating range on the contract tasks (rerun.py tables).
MAPS = {"vector": ("p40", "p50", "p60"), "spatial": ("p30", "p35", "p40")}
MODES = {"vector": ("rescale", ("none", "length")), "spatial": ("scale", ("length", "norm", "lsq"))}
DEFAULT = {"vector": "none", "spatial": "length"}


def with_mode(kv: dict, method: str, mode: str) -> dict:
    """``kv`` (a selection's calibration_kv) with the scale option on every step; the default writes none."""
    key = MODES[method][0]
    out = copy.deepcopy(kv)
    for step in out["pipeline"]:
        step.pop(key, None)
        if mode != DEFAULT[method]:
            step[key] = mode
    return out


def run(args, name: str) -> None:
    from engine.layer_select.joint_kl import calibration_set
    from engine.layer_select.joint_kl.pool import ScorerPool
    from engine.layer_select.joint_kl.search import kl_mean
    from engine.kv_compress import parse_kv_spec
    from engine.layer_select.slots import Slot

    model = MODELS[name]
    rows, meta = calibration_set.load(calibration_dir(model), per_domain=args.timing or None)
    out = ROOT / OUT / ("scale_check" + (f"_timing_{args.timing}" if args.timing else "")) / f"{name}.json"
    if not rows:
        raise SystemExit(f"{name}: no calibration examples loaded")
    state = json.loads(out.read_text()) if out.exists() else {"examples": len(rows), "kl": {}}
    devices = _devices()
    for method in args.methods:
        folder = ROOT / OUT / name / method
        reference = json.loads((folder / "singles.json").read_text())["kl"]["reference"]
        for mode in MODES[method][1]:
            todo = [tag for tag in MAPS[method] if f"{method}/{tag}/{mode}" not in state["kl"]]
            if not todo:
                continue
            # One pool per (method, mode): the scorer builds a map's spec from one template, so the
            # option rides on the template and each map is scored through the same path as rerun.py.
            selections = {tag: json.loads((folder / f"selected_{tag}.json").read_text()) for tag in todo}
            for tag, s in selections.items():
                parse_kv_spec(with_mode(s["calibration_kv"], method, mode))  # validate before loading
            template = with_mode(next(iter(selections.values()))["calibration_kv"], method, mode)
            template = {"pipeline": [template["pipeline"][0]]}
            pool = ScorerPool(devices, rows, meta, template, model=model.pretrained)
            try:
                if "dense" not in state:
                    dense = pool({"dense": {}})["dense"]
                    if any(abs(v) > 1e-6 for v in dense.values()):
                        raise RuntimeError(f"dense-vs-dense KL is not 0: {dense}")
                    state["dense"] = dense
                for tag in todo:
                    assignment = {Slot(a["layer"], a["target"]): a["level"]
                                  for a in selections[tag]["assignment"]}
                    started = time.time()
                    raw = pool({tag: assignment})[tag]
                    state["kl"][f"{method}/{tag}/{mode}"] = {
                        "kl": raw, "kl_mean": kl_mean(raw, reference)[0],
                        "compression": selections[tag]["compression"], "seconds": time.time() - started}
                    _write_atomic(out, json.dumps(state, indent=1))
                    print(f"[scale] {name} {method} {tag} {mode}: kl_mean "
                          f"{state['kl'][f'{method}/{tag}/{mode}']['kl_mean']:.4f}  "
                          f"({time.time() - started:.0f} s)", flush=True)
            finally:
                pool.close()
    table(name, state)


def table(name: str, state: dict) -> None:
    print(f"\n{name}: kl_mean (lower is better; per domain KL / reference map's, mean of 4 domains)")
    for method, (_, modes) in MODES.items():
        print(f"  {method:8s}" + "".join(f"{m:>10s}" for m in modes))
        for tag in MAPS[method]:
            cells = [state["kl"].get(f"{method}/{tag}/{m}", {}).get("kl_mean") for m in modes]
            print(f"  {tag:8s}" + "".join(f"{c:10.4f}" if c is not None else f"{'—':>10s}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=list(MAPS), choices=list(MAPS))
    parser.add_argument("--timing", type=int, default=0, help="examples per domain (own output file)")
    args = parser.parse_args()
    for name in args.models:
        run(args, name)


if __name__ == "__main__":
    main()
