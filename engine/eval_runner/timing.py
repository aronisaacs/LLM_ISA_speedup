"""Quiet, best-effort wall-clock telemetry, separate from scientific results."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

from engine.eval_runner.files import write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class TimingRecord:
    def __init__(self, root, *, name, run, worker, identity, enabled=True):
        self.enabled = enabled
        self.path = Path(root) / 'timings' / f'{os.getpid()}_{uuid4().hex}.json'
        self.started = time.monotonic()
        self.data = dict(name=name, run=str(run), worker=worker, simulation=identity,
                         host=platform.node(), machine=platform.machine(), pid=os.getpid(),
                         slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                         visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         cpu_threads=os.environ.get('OMP_NUM_THREADS'),
                         started_at=utc_now(), status='running', phases={})

    def save(self):
        self.data['elapsed_seconds'] = time.monotonic() - self.started
        if not self.enabled:
            return
        try:
            write_json(self.path, self.data)
        except OSError:
            # Optional telemetry must never abort an evaluation.
            pass

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, kind, error, traceback):
        self.data.update(status='failed' if kind else 'complete', finished_at=utc_now())
        if kind:
            self.data['error_type'] = kind.__name__
        self.save()

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        self.data['active_phase'] = name
        self.save()
        try:
            yield
        finally:
            self.data['phases'][name] = time.monotonic() - started
            self.data.pop('active_phase', None)
            self.save()
