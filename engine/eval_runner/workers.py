"""Run GPU workers with failure propagation and child-process cleanup."""
import os
import subprocess
import time


def run_workers(devices, command_for_worker):
    processes = []
    try:
        for worker, device in enumerate(devices):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
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

