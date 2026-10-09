"""Run GPU workers with failure propagation and child-process cleanup."""
import os
import subprocess
import threading
import time

# A launcher that sets this to its own pid makes the child exit when that launcher is gone,
# so a killed test or coordinator cannot leave an orphaned evaluation running for days.
PARENT_ENV = "LLM_ISA_PARENT_PID"


def exit_with_parent(poll_seconds=2.):
    """Exit this process once the pid in ``PARENT_ENV`` is no longer its parent."""
    expected = os.environ.get(PARENT_ENV)
    if not expected:
        return

    def watch():
        while os.getppid() == int(expected):
            time.sleep(poll_seconds)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="exit-with-parent").start()


def run_workers(devices, command_for_worker):
    processes = []
    try:
        for worker, device in enumerate(devices):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, **{PARENT_ENV: str(os.getpid())})
            command = command_for_worker(worker, len(devices))
            processes.append(subprocess.Popen(command, env=env))
        while any(process.poll() is None for process in processes):
            failed = next((process for process in processes if process.poll() not in (None, 0)), None)
            if failed is not None:
                raise subprocess.CalledProcessError(failed.returncode, failed.args)
            time.sleep(.5)
        for process in processes:
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

