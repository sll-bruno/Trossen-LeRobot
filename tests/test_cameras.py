import time
import unittest

from trossen_experiment.cameras import CameraError, CameraProcess
from trossen_experiment.config import CameraConfig


class CameraTests(unittest.TestCase):
    def test_camera_process_reports_frozen_stream(self):
        config = CameraConfig(
            "cam_high",
            "tests.helpers:static_camera_factory",
            0,
            4,
            3,
            30.0,
            0.1,
            2,
        )
        cameras = CameraProcess((config,))
        cameras.start()
        try:
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    cameras.latest(timeout_s=0.1)
                except CameraError as error:
                    self.assertIn("identical frames", str(error))
                    break
                if time.monotonic() >= deadline:
                    self.fail("frozen camera worker did not fail")
        finally:
            cameras.close()


if __name__ == "__main__":
    unittest.main()
