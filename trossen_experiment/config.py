"""Versioned experiment configuration with fail-closed hardware gates."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
SIDES = ("left", "right")
JOINTS_PER_ARM = 7
JOINT_NAMES = tuple(f"{side}_joint_{index}" for side in SIDES for index in range(JOINTS_PER_ARM))
CONTACT_NAMES = ("left_pad_0", "left_pad_1", "right_pad_0", "right_pad_1")


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    name: str
    driver: str
    device: str | int
    width: int
    height: int
    fps: float
    max_age_s: float
    max_identical_frames: int


@dataclass(frozen=True)
class RobotConfig:
    factory: str | None
    motion_enabled: bool
    calibration_id: str | None
    joint_names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    max_velocity: tuple[float, ...]


@dataclass(frozen=True)
class SafetyConfig:
    max_state_age_s: float
    trajectory_validator: str | None
    interpolation_step_s: float


@dataclass(frozen=True)
class TaskConfig:
    name: str
    lift_clearance_m: float
    success_hold_s: float
    require_contact: str
    pilot_horizon_s: float


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    control_hz: float
    output_dir: Path
    robot: RobotConfig
    safety: SafetyConfig
    task: TaskConfig
    cameras: tuple[CameraConfig, ...]
    positions: dict[str, tuple[str, ...]]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ConfigurationError("control_hz must be positive and finite")
        if self.robot.joint_names != JOINT_NAMES:
            raise ConfigurationError(f"joint_names must exactly match {JOINT_NAMES}")
        vectors = (
            self.robot.lower_limits,
            self.robot.upper_limits,
            self.robot.max_velocity,
        )
        if any(len(vector) != len(JOINT_NAMES) for vector in vectors):
            raise ConfigurationError("all joint limit vectors must contain 14 values")
        for index, (low, high, velocity) in enumerate(zip(*vectors)):
            if not all(math.isfinite(value) for value in (low, high, velocity)):
                raise ConfigurationError(f"joint {index} has a non-finite limit")
            if low >= high or velocity <= 0:
                raise ConfigurationError(f"joint {index} has invalid limits or velocity")
        camera_names = [camera.name for camera in self.cameras]
        if set(camera_names) != {"cam_high", "cam_front"} or len(camera_names) != 2:
            raise ConfigurationError("cameras must define exactly cam_high and cam_front")
        for camera in self.cameras:
            if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0 or camera.max_age_s <= 0:
                raise ConfigurationError(f"camera {camera.name!r} has invalid dimensions, fps, or age")
            if camera.max_identical_frames <= 0:
                raise ConfigurationError(f"camera {camera.name!r} max_identical_frames must be positive")
        if (
            not math.isfinite(self.safety.max_state_age_s)
            or self.safety.max_state_age_s <= 0
            or not math.isfinite(self.safety.interpolation_step_s)
            or self.safety.interpolation_step_s <= 0
        ):
            raise ConfigurationError("safety ages and interpolation step must be positive and finite")
        allowed_splits = {"train", "development", "test"}
        if set(self.positions) != allowed_splits:
            raise ConfigurationError(f"positions must define exactly {sorted(allowed_splits)}")
        all_positions = [position for values in self.positions.values() for position in values]
        if len(all_positions) != len(set(all_positions)):
            raise ConfigurationError("position identifiers must not occur in more than one split")
        expected_counts = {"train": 5, "development": 2, "test": 5}
        if any(len(self.positions[name]) != count for name, count in expected_counts.items()):
            raise ConfigurationError("positions must contain exactly 5 train, 2 development, and 5 test")
        if not math.isclose(self.task.lift_clearance_m, 0.05, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationError("the preserved ball task requires lift_clearance_m=0.05")
        if self.task.require_contact != "any":
            raise ConfigurationError("the preserved ball task requires contact with any pad")
        if self.task.success_hold_s != 0.0:
            raise ConfigurationError("the preserved ball task requires instantaneous success (hold_s=0)")
        if not math.isfinite(self.task.pilot_horizon_s) or self.task.pilot_horizon_s <= 0:
            raise ConfigurationError("task.pilot_horizon_s must be positive and finite")
        if self.robot.motion_enabled:
            if not self.robot.factory:
                raise ConfigurationError("robot.factory is required when physical motion is enabled")
            if not self.robot.calibration_id:
                raise ConfigurationError("calibration_id is required when physical motion is enabled")
            if not self.safety.trajectory_validator:
                raise ConfigurationError(
                    "safety.trajectory_validator is required when physical motion is enabled"
                )

    def validate_position(self, position_id: str, split: str) -> None:
        if split not in self.positions:
            raise ConfigurationError(f"unknown split {split!r}")
        if position_id not in self.positions[split]:
            actual = next(
                (name for name, values in self.positions.items() if position_id in values), None
            )
            if actual is None:
                raise ConfigurationError(f"unknown physical position {position_id!r}")
            raise ConfigurationError(
                f"position {position_id!r} belongs to split {actual!r}, not {split!r}"
            )


def import_callable(path: str) -> Callable[..., Any]:
    try:
        module_name, attribute = path.split(":", 1)
        value = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ImportError, AttributeError) as error:
        raise ConfigurationError(f"cannot import callable {path!r}: {error}") from error
    if not callable(value):
        raise ConfigurationError(f"configured value {path!r} is not callable")
    return value


def _float_tuple(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field} must be a JSON list")
    return tuple(float(item) for item in value)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        robot = raw["robot"]
        safety = raw["safety"]
        task = raw["task"]
        config = ExperimentConfig(
            schema_version=int(raw["schema_version"]),
            control_hz=float(raw["control_hz"]),
            output_dir=(source.parent / raw["output_dir"]).resolve(),
            robot=RobotConfig(
                factory=robot.get("factory"),
                motion_enabled=bool(robot.get("motion_enabled", False)),
                calibration_id=robot.get("calibration_id"),
                joint_names=tuple(robot["joint_names"]),
                lower_limits=_float_tuple(robot["lower_limits"], "robot.lower_limits"),
                upper_limits=_float_tuple(robot["upper_limits"], "robot.upper_limits"),
                max_velocity=_float_tuple(robot["max_velocity"], "robot.max_velocity"),
            ),
            safety=SafetyConfig(
                max_state_age_s=float(safety.get("max_state_age_s", 0.1)),
                trajectory_validator=safety.get("trajectory_validator"),
                interpolation_step_s=float(safety.get("interpolation_step_s", 0.02)),
            ),
            task=TaskConfig(
                name=str(task["name"]),
                lift_clearance_m=float(task["lift_clearance_m"]),
                success_hold_s=float(task["success_hold_s"]),
                require_contact=str(task["require_contact"]),
                pilot_horizon_s=float(task["pilot_horizon_s"]),
            ),
            cameras=tuple(
                CameraConfig(
                    name=str(item["name"]),
                    driver=str(item["driver"]),
                    device=item["device"],
                    width=int(item["width"]),
                    height=int(item["height"]),
                    fps=float(item["fps"]),
                    max_age_s=float(item.get("max_age_s", 0.1)),
                    max_identical_frames=int(item.get("max_identical_frames", 30)),
                )
                for item in raw["cameras"]
            ),
            positions={key: tuple(value) for key, value in raw["positions"].items()},
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid configuration {source}: {error}") from error
    config.validate()
    return config
