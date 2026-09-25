"""CPU tests for splitting a run across visible CUDA devices."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import engine.multi_run as multi_run


class WorkerSliceTests(unittest.TestCase):
    def test_three_workers_cover_every_configuration_once(self):
        configurations = [{"name": f"c{index}"} for index in range(8)]
        slices = [
            multi_run.configurations_for_worker(configurations, worker, 3) for worker in range(3)
        ]
        indexes = [index for sl in slices for index, _configuration in sl]
        self.assertEqual(sorted(indexes), list(range(1, 9)))
        self.assertEqual([index for index, _configuration in slices[0]], [1, 4, 7])
        self.assertEqual([configuration["name"] for _index, configuration in slices[1]], ["c1", "c4", "c7"])

    def test_one_worker_keeps_the_whole_list(self):
        configurations = [{"name": "only"}]
        self.assertEqual(
            multi_run.configurations_for_worker(configurations, 0, 1),
            [(1, {"name": "only"})],
        )


class VisibleDeviceTests(unittest.TestCase):
    def test_cuda_visible_devices_is_the_worker_list(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0, 2, 3"}):
            self.assertEqual(multi_run.visible_cuda_devices(), ["0", "2", "3"])

    def test_empty_cuda_visible_devices_uses_no_gpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(multi_run.visible_cuda_devices(), [])

    def test_unset_cuda_visible_devices_counts_torch_devices(self):
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, env, clear=True), patch.object(multi_run, "_cuda_device_count", return_value=4):
            self.assertEqual(multi_run.visible_cuda_devices(), ["0", "1", "2", "3"])


class WorkerCommandTests(unittest.TestCase):
    def test_command_pins_a_slice_and_keeps_skip_existing(self):
        command = multi_run.worker_command("study.json", 1, 4, True)
        self.assertIn("--worker", command)
        self.assertIn("1", command)
        self.assertIn("--workers", command)
        self.assertIn("4", command)
        self.assertIn("--skip-existing", command)
        self.assertIn("study.json", command)


if __name__ == "__main__":
    unittest.main()
