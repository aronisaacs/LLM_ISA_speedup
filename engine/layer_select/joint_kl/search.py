"""Joint greedy rung climb scored by normalised KL to dense, with lazy rounds.

Start dense. Each round, every eligible slot's next rung is scored with the whole current map
applied, and the step with the smallest ``kl_mean`` increase per unit of saving is kept. A
budget's map is saved as soon as the mean saving reaches it (as in ``rank_fill``), so one climb
gives nested maps for every budget.

``kl_mean`` = mean over domains of (domain KL / domain KL of the reference map). The reference
(a uniform map, measured once) puts the domains on one scale.

Lazy rounds (``lazy_top`` > 0): only the slots with the best last-measured estimate, plus slots
never measured at their current rung, are scored; every ``full_every``-th round scores all.
``lazy_top`` = 0 scores every slot every round.

No torch here: ``score(candidates)`` maps {label: assignment} to {label: {domain: KL}}.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from engine.layer_select.apply import kv_for_assignment, method_template
from engine.layer_select.rungs import Rung
from engine.layer_select.slots import Slot

Assignment = dict[Slot, float]
ScoreFn = Callable[[dict[str, Assignment]], dict[str, dict[str, float]]]


@dataclass
class SearchConfig:
    method_kv: dict
    rungs: tuple[Rung, ...]
    slots: tuple[Slot, ...]
    budgets: tuple[float, ...]
    reference: Assignment  # e.g. every slot at 3 bits
    lazy_top: int = 16
    full_every: int = 5
    tag_prefix: str = ""
    max_rounds: int | None = None  # timing runs stop early
    extra: dict = field(default_factory=dict)  # recorded in state and snapshots

    def identity(self) -> dict:
        """What must match for a saved state to be resumed."""
        return {
            "method_kv": method_template(self.method_kv),
            "rungs": [[r.level, r.fraction] for r in self.rungs],
            "slots": [s.tag() for s in self.slots],
            "reference": _assignment_key(self.reference),
            "lazy_top": self.lazy_top,
            "full_every": self.full_every,
            **self.extra,
        }


def kl_mean(raw: dict[str, float], reference: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Equal-weight mean of per-domain KL, each divided by the reference map's KL."""
    normed = {d: raw[d] / reference[d] for d in reference}
    return sum(normed.values()) / len(normed), normed


def run_search(score: ScoreFn, config: SearchConfig, out_dir: str | Path, log=print) -> list[dict]:
    """Climb to the largest budget; write ``selected_<tag>.json`` per budget, ``history.jsonl``, ``state.json``.

    Resumes from ``state.json`` when its identity matches ``config``. Returns the snapshots.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    budgets = sorted(config.budgets)
    top = max(r.fraction for r in config.rungs)
    if budgets[-1] > top + 1e-12:
        raise ValueError(f"budget {budgets[-1]} exceeds the top rung's saving {top}")

    if state_path.exists():
        st = json.loads(state_path.read_text())
        if st["identity"] != json.loads(json.dumps(config.identity())):
            raise ValueError(f"{state_path} was made with other settings; move it away to start over")
        index = {Slot.from_dict(r): r["index"] for r in st["index"]}
        est = {Slot.from_dict(r): r["est"] for r in st["est"]}
        cache = st["cache"]
        rnd, done = st["round"], st["done"]
        reference_kl = st["reference_kl"]
        log(f"resumed at round {rnd}, saving {_saving(index, config):.4f}")
    else:
        index = {slot: -1 for slot in config.slots}
        est = {slot: None for slot in config.slots}
        cache, rnd, done = {}, 0, []
        reference_kl = None

    def scored(cands: dict[str, Assignment]) -> dict[str, dict[str, float]]:
        todo = {}
        for assignment in cands.values():
            key = _assignment_key(assignment)
            if key not in cache:
                todo[key] = assignment
        if todo:
            fresh = score(todo)
            cache.update({key: fresh[key] for key in todo})
        return {label: cache[_assignment_key(a)] for label, a in cands.items()}

    if reference_kl is None:
        t0 = time.time()
        reference_kl = scored({"reference": config.reference})["reference"]
        log(f"reference KL {_fmt(reference_kl)} ({time.time() - t0:.0f} s)")
        if not all(v > 0 for v in reference_kl.values()):
            raise RuntimeError(f"reference map leaves some domain unchanged: {reference_kl}")

    def assignment_of(idx: dict[Slot, int]) -> Assignment:
        return {s: config.rungs[i].level for s, i in idx.items() if i >= 0}

    def current_score():
        if all(i < 0 for i in index.values()):
            return 0.0, {d: 0.0 for d in reference_kl}, {d: 0.0 for d in reference_kl}
        raw = scored({"current": assignment_of(index)})["current"]
        return (*kl_mean(raw, reference_kl), raw)

    def save_state():
        st = {
            "identity": config.identity(), "round": rnd, "done": done, "reference_kl": reference_kl,
            "index": [{**s.to_dict(), "index": i} for s, i in sorted(index.items())],
            "est": [{**s.to_dict(), "est": e} for s, e in sorted(est.items())],
            "cache": cache,
        }
        _write_atomic(state_path, json.dumps(st))

    snapshots = [json.loads((out / f"selected_{_tag(config, b)}.json").read_text()) for b in done]
    cur, cur_normed, cur_raw = current_score()
    while True:
        saving = _saving(index, config)
        pending = [b for b in budgets if b not in done]
        if pending and saving >= pending[0] - 1e-12:
            snap = _snapshot(config, pending[0], assignment_of(index), saving, rnd, cur, cur_normed, cur_raw,
                             reference_kl)
            _write_atomic(out / f"selected_{snap['tag']}.json", json.dumps(snap, indent=2) + "\n")
            snapshots.append(snap)
            done.append(pending[0])
            save_state()
            log(f"[snapshot] {snap['tag']}: saving {saving:.4f}, kl_mean {cur:.5f}, round {rnd}")
            continue
        if not pending:
            break
        if config.max_rounds is not None and rnd >= config.max_rounds:
            log(f"stopped after {rnd} rounds (max_rounds)")
            break

        eligible = [s for s in config.slots if index[s] + 1 < len(config.rungs)]
        if not eligible:
            raise RuntimeError("no slot can climb; budget unreachable")
        full = config.lazy_top <= 0 or rnd % max(1, config.full_every) == 0
        forced = [s for s in eligible if est[s] is None]
        ranked = sorted((s for s in eligible if est[s] is not None), key=lambda s: (est[s], s))
        shortlist = forced + ranked[: max(0, config.lazy_top - len(forced))]
        chosen = eligible if full else shortlist

        gain = {s.tag(): _gain(config, s, index[s]) for s in chosen}
        cands = {s.tag(): assignment_of({**index, s: index[s] + 1}) for s in chosen}
        t0 = time.time()
        raw = scored(cands)
        secs = time.time() - t0
        scores = {k: kl_mean(v, reference_kl) for k, v in raw.items()}
        value = {k: (scores[k][0] - cur) / gain[k] for k in scores}
        est_before = {s.tag(): est[s] for s in chosen}
        for s in chosen:
            est[s] = value[s.tag()]
        best_tag = min(value, key=lambda k: (value[k], k))
        best = next(s for s in chosen if s.tag() == best_tag)
        shortlist_hit = (best in shortlist) if full and config.lazy_top > 0 and rnd > 0 else None

        prev = cur
        index[best] += 1
        est[best] = None  # its next rung has not been measured
        cur, cur_normed = scores[best_tag]
        cur_raw = raw[best_tag]
        rnd += 1
        table = sorted(([k, scores[k][0], scores[k][0] - prev, gain[k], value[k], est_before[k]] for k in scores),
                       key=lambda r: (r[4], r[0]))
        _append(out / "history.jsonl", {
            "round": rnd, "type": "full" if full else "lazy", "candidates": len(cands), "seconds": secs,
            "pick": best_tag, "pick_level": config.rungs[index[best]].level, "gain": gain[best_tag],
            "kl_mean_before": prev, "kl_mean_after": cur, "normalised": cur_normed, "raw_kl": cur_raw,
            "saving": _saving(index, config),
            "margin_to_2nd": table[1][4] - table[0][4] if len(table) > 1 else None,
            "shortlist_hit": shortlist_hit,
            "table": table,  # [slot, kl_mean, increase, saving gain, increase per saving, estimate before]
        })
        log(f"[round {rnd:3d} {'full' if full else 'lazy'} n={len(cands):2d} {secs:6.1f} s] "
            f"{best_tag} -> {config.rungs[index[best]].level}  kl_mean {prev:.5f} -> {cur:.5f}  "
            f"saving {_saving(index, config):.4f}" + ("" if shortlist_hit is None else f"  hit={shortlist_hit}"))
        save_state()
    return snapshots


def _gain(config: SearchConfig, slot: Slot, current: int) -> float:
    before = config.rungs[current].fraction if current >= 0 else 0.0
    gain = config.rungs[current + 1].fraction - before
    if gain <= 0:
        raise ValueError("rung savings must increase")
    return gain


def _saving(index: dict[Slot, int], config: SearchConfig) -> float:
    return sum(config.rungs[i].fraction for i in index.values() if i >= 0) / len(config.slots)


def _tag(config: SearchConfig, budget: float) -> str:
    return f"{config.tag_prefix}p{round(budget * 100):02d}"


def _snapshot(config, budget, assignment, saving, rnd, score, normed, raw, reference_kl) -> dict:
    """Same fields as ``budgets.selections_for_budgets`` payloads, plus the calibration scores."""
    template = method_template(config.method_kv)
    return {
        "tag": _tag(config, budget),
        "budget": budget,
        "compression": saving,
        "n_layers": max(s.layer for s in config.slots) + 1,
        "levels": [r.level for r in config.rungs],
        "assignment": [{**s.to_dict(), "level": lvl} for s, lvl in sorted(assignment.items())],
        "kv": kv_for_assignment(template, assignment),
        "calibration": {
            "allocator": "joint_kl", "kl_mean": score, "normalised": normed, "raw_kl": raw,
            "reference_kl": reference_kl, "rounds": rnd, "lazy_top": config.lazy_top,
            "full_every": config.full_every, **config.extra,
        },
    }


def _assignment_key(assignment: Assignment) -> str:
    items = sorted((s.layer, s.target, float(lvl)) for s, lvl in assignment.items())
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:16]


def _fmt(raw: dict[str, float]) -> str:
    return ", ".join(f"{d} {v:.4f}" for d, v in raw.items())


def _append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
