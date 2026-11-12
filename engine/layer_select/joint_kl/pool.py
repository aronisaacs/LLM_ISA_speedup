"""One KLScorer per device, with the examples split between them.

Every worker scores every candidate on its own share of the examples (balanced by length), so
each example's dense forward runs once and all devices finish together. One device runs in this
process. Workers exit when the coordinator dies (``exit_with_parent``), and a worker failure
raises here with its traceback.
"""

from __future__ import annotations

import os
import queue
import traceback

from engine.eval_runner.workers import PARENT_ENV, exit_with_parent
from engine.layer_select.joint_kl.calibration_set import partition
from engine.layer_select.joint_kl.scorer import KLScorer, merge


class ScorerPool:
    def __init__(self, devices: list[str], rows: list[dict], meta: dict, method_kv: dict,
                 model: str | None = None, dtype: str | None = None):
        self.devices = list(devices)
        shards = partition(rows, len(self.devices))
        if len(self.devices) == 1:
            self.local = KLScorer(shards[0], meta, method_kv, self.devices[0], model=model, dtype=dtype)
            return
        import multiprocessing as mp

        self.local = None
        ctx = mp.get_context("spawn")
        self.results = ctx.Queue()
        self.inboxes = []
        self.workers = []
        os.environ[PARENT_ENV] = str(os.getpid())
        for rank, (device, shard) in enumerate(zip(self.devices, shards)):
            inbox = ctx.Queue()
            proc = ctx.Process(target=_serve, args=(rank, device, shard, meta, method_kv, model, dtype,
                                                    inbox, self.results), daemon=True)
            proc.start()
            self.inboxes.append(inbox)
            self.workers.append(proc)
        self._collect("ready")

    def __call__(self, candidates: dict[str, dict]) -> dict[str, dict[str, float]]:
        if self.local is not None:
            return merge([self.local.kl_sums(candidates)])
        for inbox in self.inboxes:
            inbox.put(candidates)
        return merge(self._collect("sums"))

    def close(self) -> None:
        if self.local is not None:
            return
        for inbox in self.inboxes:
            inbox.put(None)
        for proc in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()

    def _collect(self, kind: str) -> list:
        got: dict[int, object] = {}
        while len(got) < len(self.workers):
            try:
                rank, tag, payload = self.results.get(timeout=10)
            except queue.Empty:
                dead = [r for r, p in enumerate(self.workers) if not p.is_alive() and r not in got]
                if dead:
                    raise RuntimeError(f"scorer worker(s) {dead} exited") from None
                continue
            if tag == "error":
                raise RuntimeError(f"scorer worker {rank} on {self.devices[rank]} failed:\n{payload}")
            if tag != kind:
                raise RuntimeError(f"worker {rank}: expected {kind}, got {tag}")
            got[rank] = payload
        return [got[r] for r in range(len(self.workers))]


def _serve(rank, device, shard, meta, method_kv, model, dtype, inbox, results) -> None:
    exit_with_parent()
    try:
        scorer = KLScorer(shard, meta, method_kv, device, model=model, dtype=dtype)
        results.put((rank, "ready", None))
        while (candidates := inbox.get()) is not None:
            results.put((rank, "sums", scorer.kl_sums(candidates)))
    except BaseException:  # noqa: BLE001 - report everything to the coordinator
        results.put((rank, "error", traceback.format_exc()))
