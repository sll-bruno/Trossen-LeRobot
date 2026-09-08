from pathlib import Path

from trossen_experiment.config import (
    CameraConfig,
    ExperimentConfig,
    JOINT_NAMES,
    RobotConfig,
    SafetyConfig,
    TaskConfig,
)


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def make_config(root: Path, motion_enabled=False, trajectory_validator=None):
    config = ExperimentConfig(
        schema_version=1,
        control_hz=10.0,
        output_dir=root,
        robot=RobotConfig(
            factory="tests.helpers:fake_factory" if motion_enabled else None,
            motion_enabled=motion_enabled,
            calibration_id="cal-1" if motion_enabled else None,
            joint_names=JOINT_NAMES,
            lower_limits=(-1.0,) * 14,
            upper_limits=(1.0,) * 14,
            max_velocity=(1.0,) * 14,
        ),
        safety=SafetyConfig(0.1, trajectory_validator, 0.02),
        task=TaskConfig("lift ball", 0.05, 0.0, "any", 30.0),
        cameras=(
            CameraConfig("cam_high", "unused:factory", 0, 4, 3, 10, 0.1, 30),
            CameraConfig("cam_front", "unused:factory", 1, 4, 3, 10, 0.1, 30),
        ),
        positions={
            "train": ("train_1", "train_2", "train_3", "train_4", "train_5"),
            "development": ("dev_1", "dev_2"),
            "test": ("test_1", "test_2", "test_3", "test_4", "test_5"),
        },
    )
    return config


def fake_factory():
    raise AssertionError("hardware factory must not be called in offline tests")


def constant_policy_factory(value=0.0):
    def policy(observation):
        return (float(value),) * 14

    return policy


class StaticCameraDriver:
    def __init__(self, config):
        self.config = config

    def open(self):
        return None

    def read(self):
        import numpy as np

        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def close(self):
        return None


def static_camera_factory(config):
    return StaticCameraDriver(config)
