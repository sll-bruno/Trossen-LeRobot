import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from trossen_experiment.config import ConfigurationError, load_config
from trossen_experiment.models import RobotState
from trossen_experiment.robot import (
    LeRobotBackend,
    PartialSendError,
    RobotError,
    SafeRobot,
    SimulatedBackend,
)
from trossen_experiment.safety import SafetyValidator, SafetyViolation

from tests.helpers import Clock, make_config


class ConfigSafetyRobotTests(unittest.TestCase):
    def test_mock_configuration_is_valid(self):
        config = load_config(Path(__file__).parents[1] / "configs/experiment.mock.json")
        self.assertFalse(config.robot.motion_enabled)
        self.assertEqual(config.task.lift_clearance_m, 0.05)
        self.assertEqual(config.task.success_hold_s, 0.0)
        self.assertEqual(config.task.require_contact, "any")

    def test_physical_configuration_requires_trajectory_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory), motion_enabled=True)
            with self.assertRaisesRegex(ConfigurationError, "trajectory_validator"):
                config.validate()

    def test_position_cannot_be_relabelled_into_another_split(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with self.assertRaisesRegex(ConfigurationError, "belongs to split 'test'"):
                config.validate_position("test_1", "train")

    def test_scientific_task_contract_cannot_drift_in_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with self.assertRaisesRegex(ConfigurationError, "lift_clearance_m=0.05"):
                replace(config, task=replace(config.task, lift_clearance_m=0.06)).validate()
            with self.assertRaisesRegex(ConfigurationError, "contact with any pad"):
                replace(config, task=replace(config.task, require_contact="both_arms")).validate()

    def test_safety_checks_freshness_bounds_velocity_and_latches(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock(1.0)
            config = make_config(Path(directory))
            safety = SafetyValidator(config.robot, config.safety, clock=clock)
            state = RobotState.from_values((0.0,) * 14, 0.95)
            self.assertEqual(safety.validate_target(state, (0.05,) * 14, 0.1), (0.05,) * 14)
            with self.assertRaisesRegex(SafetyViolation, "velocity"):
                safety.validate_target(state, (0.2,) * 14, 0.1)
            with self.assertRaisesRegex(SafetyViolation, "latched"):
                safety.validate_target(state, (0.0,) * 14, 0.1)

    def test_safety_rejects_non_finite_state(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock(1.0)
            config = make_config(Path(directory))
            safety = SafetyValidator(config.robot, config.safety, clock=clock)
            state = RobotState.from_values((math.nan,) + (0.0,) * 13, 1.0)
            with self.assertRaisesRegex(SafetyViolation, "non-finite"):
                safety.validate_target(state, (0.0,) * 14, 0.1)

    def test_partial_second_arm_send_is_visible_and_latched(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock(1.0)
            config = make_config(Path(directory))
            backend = SimulatedBackend(clock=clock, fail_side="right")
            robot = SafeRobot(
                backend, SafetyValidator(config.robot, config.safety, clock=clock), clock=clock
            )
            robot.connect()
            state = robot.read_state()
            with self.assertRaises(PartialSendError) as caught:
                robot.send(state, (0.01,) * 14, 0.1)
            self.assertEqual(caught.exception.sent_sides, ("left",))
            self.assertEqual(backend.positions[:7], (0.01,) * 7)
            self.assertEqual(backend.positions[7:], (0.0,) * 7)
            self.assertIsNotNone(robot.safety.latched_reason)
            self.assertIsNotNone(backend.stopped_reason)
            self.assertEqual(caught.exception.requested_target, (0.01,) * 14)
            self.assertEqual(caught.exception.sent_target[:7], (0.01,) * 7)
            self.assertEqual(caught.exception.sent_target[7:], (0.0,) * 7)

    def test_physical_backend_rejects_uncalibrated_lifecycle(self):
        class RawRobot:
            def connect(self):
                self.connected = True

            def abort_connection(self):
                self.connected = False

        backend = LeRobotBackend(RawRobot)
        with self.assertRaisesRegex(RobotError, "calibrated lifecycle methods"):
            backend.connect()


if __name__ == "__main__":
    unittest.main()
