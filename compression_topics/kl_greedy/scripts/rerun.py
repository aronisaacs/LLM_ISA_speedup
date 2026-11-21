#!/usr/bin/env python3
"""Vector and spatial compression on Llama 3.1 8B, Qwen3.5-9B and CodeLlama 7B: greedy on 4-domain KL.

One protocol for both methods and all three models (Aron, 2026-11-21), replacing the per-slot
WikiText perplexity sweeps (vector ``model_budget_study.py``, spatial ``ladder.py``):

  calibrate  the model's 4-domain KL calibration set (WikiText, ARC 5-shot, GSM8K, MBPP; the dense
             model's own greedy answers). Llama's is ``catalog/calibration/llama31_8b_mixed4``; Qwen's
             and CodeLlama's are built here once (``joint_kl/build_calibration_set.py``) if missing.
  singles    KL(dense || compressed) on that set for every KV slot (one layer's K or V) alone at every
             level, plus a uniform reference map. Per domain, KL / reference KL; equal-weight mean
             over the 4 domains = the slot's score (``kl_mean``, as in the joint-KL search).
  greedy     from dense, repeatedly raise the slot whose next level costs the least score per unit
             of saving, until the mean saving over the KV slots reaches each budget (isolated
             greedy: scores come from the singles; no re-scoring on top of the map built so far).
  tasks      dense and every budget's map on the model's tasks (lm-eval, results.json, skip existing).
  table      accuracy by budget.

Methods (savings are net of the format's metadata, means over the model's KV slots):

  vector   per-scalar bitmask prune (``vector_compress``) at 25/50/75 % of each vector, net of the
           one-bit-per-value mask; budgets 10/20/30/40/50 %; reference map: every slot at 37 %
           (30.5 % net, as E11's uniform 30 %). Calibration compresses every token; the task runs
           keep the newest 32 tokens exact (``recent_window``; Aron, 2026-11-21).
  spatial  spatial pair compression with query-aware keys, 32-token windows, dense/PR0/PR16/PR32;
           slot savings 10/25/40/47/49 % (net of the 2-bit pair tags); budgets 10/20/30/35/40 %;
           reference map: every slot at 30 %. Query weights: the model's official calibration.

Models and tasks: Llama 3.1 8B-Instruct and Qwen3.5-9B (thinking off; only its 8 full-attention
layers have a KV cache) on C-Eval 5-shot, GSM8K and HumanEval-instruct; CodeLlama 7B (base) on
HumanEval with its body-indent prompt.

Everything resumes: singles are saved after every chunk of candidates, task rows already in
results.json are skipped. Outputs: ``compression_topics/kl_greedy/results/<model>/<method>/``.

  python compression_topics/kl_greedy/scripts/rerun.py                          # everything
  python compression_topics/kl_greedy/scripts/rerun.py --models qwen --methods spatial
  python compression_topics/kl_greedy/scripts/rerun.py --through greedy         # no tasks yet
  python compression_topics/kl_greedy/scripts/rerun.py --timing 4 --models llama  # 4 ex/domain, own folder

GPUs: one KL scorer per visible GPU; tasks go through engine/multi_run.py (one worker per GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import vector_compress  # noqa: E402
from catalog.models import BATCH_SIZE, CODELLAMA_7B, LLAMA31_8B, QWEN35_9B  # noqa: E402
from catalog.tasks import CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_CODELLAMA, HUMANEVAL_INSTRUCT  # noqa: E402
from compression_topics.spatial.algorithms.formats import OFFICIAL_WINDOW, dense_bits  # noqa: E402
from compression_topics.vector.bitmask import net_saving  # noqa: E402
from engine.layer_select.apply import kv_for_assignment, method_template  # noqa: E402
from engine.layer_select.slots import Slot  # noqa: E402

STAGES = ("calibrate", "singles", "greedy", "tasks", "table")
VECTOR_TASK_WINDOW = 32  # newest tokens kept exact in the task runs only, not in calibration
OUT = Path("compression_topics/kl_greedy/results")  # relative to ROOT, so run files work anywhere
CALIBRATION = Path("catalog/calibration")
TASK_ORDER = ("gsm8k", "humaneval_instruct", "humaneval", "ceval")  # slowest first
METRICS = {
    "ceval-valid": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval_instruct": "pass@1,create_test",
    "humaneval_codellama": "pass@1,create_test",
}


@dataclass(frozen=True)
class Model:
    model_args: str
    pretrained: str
    head_dim: int
    layers: int
    kv_layers: tuple[int, ...]
    tasks: tuple[dict, ...]
    calibration: str  # folder under catalog/calibration
    build: tuple[str, ...]  # build_calibration_set.py arguments when the folder is missing
    query_weights: str  # spatial: calibrated E[q^2], relative to ROOT


CHAT_TASKS = (CEVAL_VALID_5SHOT, GSM8K_20PCT, HUMANEVAL_INSTRUCT)
MODELS = {
    "llama": Model(LLAMA31_8B, "meta-llama/Llama-3.1-8B-Instruct", 128, 32, tuple(range(32)), CHAT_TASKS,
                   "llama31_8b_mixed4", (),
                   "compression_topics/spatial/figures/official_llama/query_weights.pt"),
    "qwen": Model(QWEN35_9B, "Qwen/Qwen3.5-9B", 256, 32, tuple(range(3, 32, 4)), CHAT_TASKS,
                  "qwen35_9b_mixed4",
                  ("--prompt-format", "chat", "--gsm8k-shots", "5", "--mbpp-shots", "3",
                   "--chat-template-kwargs", '{"enable_thinking": false}'),
                  "compression_topics/spatial/figures/official_qwen/query_weights.pt"),
    "codellama": Model(CODELLAMA_7B, "codellama/CodeLlama-7b-hf", 128, 32, tuple(range(32)),
                       (HUMANEVAL_CODELLAMA,), "codellama_7b_mixed4", ("--prompt-format", "plain"),
                       "compression_topics/spatial/figures/official_codellama/query_weights.pt"),
}


@dataclass(frozen=True)
class Method:
    levels: tuple[int, ...]  # per-slot levels, mildest first
    budgets: tuple[float, ...]
    reference: int  # level of every slot in the reference map
    batch_size: int


METHODS = {
    "vector": Method((25, 50, 75), (0.10, 0.20, 0.30, 0.40, 0.50), 37, BATCH_SIZE),
    # Spatial compression needs batch size one.
    "spatial": Method((10, 25, 40, 47, 49), (0.10, 0.20, 0.30, 0.35, 0.40), 30, 1),
}


def template(method: str, model: Model, *, tasks: bool = False) -> dict:
    """The method's kv with empty layer lists; the level is set per slot.

    ``tasks``: the task runs' variant (vector keeps the newest 32 tokens exact there).
    """
    if method == "vector":
        return method_template(vector_compress(prune_pct=25, recent_window=VECTOR_TASK_WINDOW if tasks else 0))
    from compression_topics.spatial.algorithms.methods import SpatialConfig

    digest = hashlib.sha256((ROOT / model.query_weights).read_bytes()).hexdigest()
    config = SpatialConfig(query_weights=model.query_weights, query_weights_sha256=digest, window=OFFICIAL_WINDOW)
    return method_template(config.pipeline(0.25))


def saving(method: str, model: Model, slot: Slot, level: int) -> float:
    """Fraction of the slot's dense bits saved at ``level``, net of the format's metadata."""
    if level <= 0:
        return 0.0
    if method == "vector":
        return net_saving("vector_compress", slot.target, level, model.head_dim)
    from compression_topics.spatial.algorithms.windows import block_bits

    dense = dense_bits(OFFICIAL_WINDOW, model.head_dim)
    return 1 - block_bits(model.head_dim, level / 100, OFFICIAL_WINDOW) / dense


def slots(model: Model) -> list[Slot]:
    return [Slot(layer, target) for layer in model.kv_layers for target in ("k", "v")]


def out_dir(args, name: str, method: str) -> Path:
    base = OUT / (f"timing_{args.timing}" if args.timing else "") / name / method
    return ROOT / base


def calibration_dir(model: Model) -> Path:
    return ROOT / CALIBRATION / model.calibration


# --------------------------------------------------------------------------- calibrate

def calibrate(name: str, model: Model) -> None:
    folder = calibration_dir(model)
    if (folder / "manifest_metadata.json").exists():
        meta = json.loads((folder / "manifest_metadata.json").read_text())
        if meta["model"] != model.pretrained:
            raise SystemExit(f"{folder} is for {meta['model']}, not {model.pretrained}")
        print(f"[calibrate] {name}: {folder.relative_to(ROOT)} exists", flush=True)
        return
    if not model.build:
        raise SystemExit(f"{folder} is missing and has no build recipe")
    command = [sys.executable, "-u", str(ROOT / "engine/layer_select/joint_kl/build_calibration_set.py"),
               "--model", model.pretrained, "--output-dir", str(folder), *model.build]
    print(f"[calibrate] {name}: building {folder.relative_to(ROOT)} (1 GPU)", flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=_one_gpu_env())
    (folder / "README.md").write_text(
        f"# {model.pretrained}, 4-domain KL calibration set\n\n"
        "Built by `engine/layer_select/joint_kl/build_calibration_set.py` (from "
        "`compression_topics/kl_greedy/scripts/rerun.py`) with "
        f"`{' '.join(model.build)}`. Same domains, sources, selection and format as "
        "`llama31_8b_mixed4`; the scored answers are this model's own dense greedy answers. "
        "`manifest_metadata.json` holds its SHA-256, checked on load.\n")


def _one_gpu_env() -> dict:
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    return env


# --------------------------------------------------------------------------- singles

def candidates(method: str, model: Model) -> dict[str, dict[Slot, int]]:
    """Reference map, then every slot alone at every level (dense = the empty map, checked apart)."""
    spec = METHODS[method]
    found = {"reference": {slot: spec.reference for slot in slots(model)}}
    for slot in slots(model):
        for level in spec.levels:
            found[f"{slot.tag()}_{level}"] = {slot: level}
    return found


def singles(args, name: str, method: str) -> None:
    from engine.layer_select.joint_kl import calibration_set

    model = MODELS[name]
    rows, meta = calibration_set.load(calibration_dir(model), per_domain=args.timing)
    if meta["model"] != model.pretrained:
        raise SystemExit(f"calibration set is for {meta['model']}, not {model.pretrained}")
    kv = template(method, model)
    identity = {
        "model": model.pretrained, "method_kv": kv, "levels": list(METHODS[method].levels),
        "reference_level": METHODS[method].reference, "kv_layers": list(model.kv_layers),
        "manifest_sha256": meta["manifest_sha256"], "examples": len(rows), "score": "kl_dense_vs_compressed",
    }
    out = out_dir(args, name, method)
    path = out / "singles.json"
    state = json.loads(path.read_text()) if path.exists() else {"identity": identity, "kl": {}}
    if state["identity"] != json.loads(json.dumps(identity)):
        raise SystemExit(f"{path} was made with other settings; move it away to start over")
    todo = {label: a for label, a in candidates(method, model).items() if label not in state["kl"]}
    if not todo:
        print(f"[singles] {name}/{method}: all {len(state['kl'])} scored", flush=True)
        return

    from engine.layer_select.joint_kl.pool import ScorerPool

    devices = _devices()
    print(f"[singles] {name}/{method}: {len(todo)} of {len(state['kl']) + len(todo)} candidates to score on "
          f"{len(rows)} examples, devices {devices}", flush=True)
    pool = ScorerPool(devices, rows, meta, kv, model=model.pretrained)
    try:
        dense = pool({"dense": {}})["dense"]  # dense against itself: must be 0
        if any(abs(v) > 1e-6 for v in dense.values()):
            raise RuntimeError(f"dense-vs-dense KL is not 0: {dense}")
        labels = list(todo)
        started, done = time.time(), 0
        for first in range(0, len(labels), args.chunk):
            chunk = {label: todo[label] for label in labels[first:first + args.chunk]}
            state["kl"].update(pool(chunk))
            _write_atomic(path, json.dumps(state, indent=1))
            done += len(chunk)
            rate = (time.time() - started) / done
            print(f"[singles] {name}/{method}: {done}/{len(labels)}  {rate:.1f} s/candidate  "
                  f"eta {rate * (len(labels) - done) / 60:.0f} min", flush=True)
    finally:
        pool.close()
    reference = state["kl"]["reference"]
    if not all(v > 0 for v in reference.values()):
        raise RuntimeError(f"reference map leaves some domain unchanged: {reference}")


def _devices() -> list[str]:
    import torch

    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["mps" if torch.backends.mps.is_available() else "cpu"]


# --------------------------------------------------------------------------- greedy

def scores(kl: dict[str, dict[str, float]]) -> dict[str, float]:
    """kl_mean per candidate: per domain KL / reference KL, equal-weight mean over the domains."""
    from engine.layer_select.joint_kl.search import kl_mean

    reference = kl["reference"]
    return {label: kl_mean(raw, reference)[0] for label, raw in kl.items()}


def allocate(method: str, model: Model, kl: dict[str, dict[str, float]], budget: float) -> dict:
    from engine.layer_select.greedy.calibrated import choose_candidates

    spec = METHODS[method]
    score = scores(kl)
    choices = {}
    for slot in slots(model):
        choices[(slot.layer, slot.target)] = [
            {"level": level, "allocation_saving": saving(method, model, slot, level),
             "kl_mean": score[f"{slot.tag()}_{level}"]}
            for level in spec.levels]
    selected, total = choose_candidates(choices, 0.0, budget, score="kl_mean")
    assignment = {Slot(layer, target): row["level"] for (layer, target), row in selected.items()}
    n = len(choices)
    return {
        "tag": f"p{round(budget * 100):02d}",
        "budget": budget,
        "compression": total / n,  # net saving, mean over the KV slots
        "levels": list(spec.levels),
        "kv_layers": list(model.kv_layers),
        "assignment": [{**s.to_dict(), "level": lvl} for s, lvl in sorted(assignment.items())],
        "kv": kv_for_assignment(template(method, model, tasks=True), assignment),  # what the tasks run
        "calibration_kv": kv_for_assignment(template(method, model), assignment),  # what the KL scored
        "calibration": {
            "allocator": "greedy_isolated_kl", "score": "kl_mean (4 domains, / reference map)",
            "sum_of_slot_scores": sum(score[f"{s.tag()}_{lvl}"] for s, lvl in assignment.items()),
            "reference_level": spec.reference, "reference_kl": kl["reference"],
        },
    }


def greedy(args, name: str, method: str) -> list[dict]:
    out = out_dir(args, name, method)
    state = json.loads((out / "singles.json").read_text())
    model = MODELS[name]
    missing = [label for label in candidates(method, model) if label not in state["kl"]]
    if missing:
        raise SystemExit(f"{out / 'singles.json'} is missing {len(missing)} candidates; run singles first")
    selections = [allocate(method, model, state["kl"], b) for b in METHODS[method].budgets]
    for selection in selections:
        _write_atomic(out / f"selected_{selection['tag']}.json", json.dumps(selection, indent=2) + "\n")
    _write_atomic(out / "selections.json", json.dumps(selections, indent=2) + "\n")
    print(f"[greedy] {name}/{method}:  budget    net   K levels           V levels", flush=True)
    for s in selections:
        count = lambda t: "/".join(  # noqa: E731
            str(sum(1 for a in s["assignment"] if a["target"] == t and a["level"] == lvl))
            for lvl in METHODS[method].levels)
        print(f"   {s['tag']:>6} {s['compression']:7.4f}   {count('k'):<18} {count('v')}", flush=True)
    return selections


# --------------------------------------------------------------------------- tasks

def tasks_run(args, name: str, method: str) -> dict:
    from engine.layer_select.budgets import budget_run

    model, spec = MODELS[name], METHODS[method]
    out = out_dir(args, name, method)
    selections = json.loads((out / "selections.json").read_text())
    results_dir = str(out.relative_to(ROOT) / "tasks")
    run = budget_run(selections, tasks=model.tasks, results_dir=results_dir)
    run.update(model="hf", model_args=model.model_args, batch_size=spec.batch_size)
    for configuration in run["configurations"]:
        configuration["name"] = configuration["name"].replace("llama31_", f"{name}_{method}_", 1)
    rank = lambda c: next(i for i, t in enumerate(TASK_ORDER) if c["tasks"][0].startswith(t))  # noqa: E731
    run["configurations"].sort(key=lambda c: (rank(c), not c["name"].endswith("_dense")))
    _write_atomic(out / "tasks_run.json", json.dumps(run, indent=2) + "\n")
    return run


def tasks(args, name: str, method: str) -> None:
    run = tasks_run(args, name, method)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(run, handle)
        path = Path(handle.name)
    try:
        subprocess.run([sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path),
                        "--skip-existing"], cwd=ROOT, check=True, env=dict(os.environ))
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- table

def table(args, name: str, method: str) -> None:
    from engine.eval_runner.cache import simulation_identity
    from engine.eval_runner.index import find_result
    from engine.kv_compress import parse_kv_spec

    out = out_dir(args, name, method)
    path = out / "tasks_run.json"
    if not path.exists():
        print(f"[table] {name}/{method}: no tasks_run.json yet", flush=True)
        return
    run = json.loads(path.read_text())
    base = {key: value for key, value in run.items() if key != "configurations"}
    tags = ["dense"] + [s["tag"] for s in json.loads((out / "selections.json").read_text())]
    net = {s["tag"]: s["compression"] for s in json.loads((out / "selections.json").read_text())}
    net["dense"] = 0.0
    names = [t["name_task"] for t in MODELS[name].tasks]
    cells = {}
    for configuration in run["configurations"]:
        task_key = configuration["tasks"][0]
        row = find_result(simulation_identity(base, configuration, parse_kv_spec(configuration["kv"])))
        task = next(t["name_task"] for t in MODELS[name].tasks if t["tasks"][0] == task_key)
        tag = configuration["output_path"].rsplit("_", 1)[1].removesuffix(".json")
        cells[(tag, task)] = None if row is None else row["scores"][task_key][METRICS[task_key]]
    lines = [f"{name}/{method}" + "".join(f"{n:>20}" for n in names) + f"{'net saving':>12}"]
    for tag in tags:
        values = [cells.get((tag, n)) for n in names]
        lines.append(f"{tag:<14}" + "".join(f"{'—':>20}" if v is None else f"{100 * v:19.1f}%" for v in values)
                     + f"{100 * net[tag]:11.1f}%")
    text = "\n".join(lines)
    print(text, flush=True)
    (out / "table.txt").write_text(text + "\n")


# --------------------------------------------------------------------------- main

def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    p.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    p.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    p.add_argument("--through", choices=STAGES, default=STAGES[-1])
    p.add_argument("--chunk", type=int, default=24, help="candidates scored between saves")
    p.add_argument("--timing", type=int, metavar="N",
                   help="N calibration examples per domain, own results folder, no tasks")
    return p


def main() -> None:
    args = parser().parse_args()
    os.chdir(ROOT)  # run files and the spatial query-weight paths are relative to the repo
    first, last = STAGES.index(args.start), STAGES.index(args.through)
    if first > last:
        raise SystemExit(f"--from {args.start} is after --through {args.through}")
    stages = STAGES[first:last + 1]
    if args.timing:
        stages = tuple(s for s in stages if s not in ("tasks", "table"))
    from engine.eval_runner.progress import mirror_terminal

    mirror_terminal(ROOT)
    # Stage-major: every KL scoring (GPU-heavy, short) first, then all task runs.
    for stage in stages:
        for name in args.models:
            if stage == "calibrate":
                calibrate(name, MODELS[name])
                continue
            for method in args.methods:
                print(f"=== {stage} {name}/{method} ===", flush=True)
                {"singles": singles, "greedy": greedy, "tasks": tasks, "table": table}[stage](args, name, method)


if __name__ == "__main__":
    main()
