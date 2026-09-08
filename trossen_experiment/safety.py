"""Fail-closed validation shared by every source of robot actions."""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence

from .config import RobotConfig, SafetyConfig
from .models import RobotState


class SafetyViolation(RuntimeError):
    pass


TrajectoryValidator = Callable[[tuple[float, ...]], bool | None]


class SafetyValidator:
    def __init__(
        self,
        robot: RobotConfig,
        safety: SafetyConfig,
        trajectory_validator: TrajectoryValidator | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.trajectory_validator = trajectory_validator
        self.clock = clock
        self._latched_reason: str | None = None

    @property
    def latched_reason(self) -> str | None:
        return self._latched_reason

    def latch(self, reason: str) -> None:
        self._latched_reason = reason

    def clear(self, state: RobotState) -> None:
        self._validate_state(state)
        self._latched_reason = None

    def _fail(self, reason: str) -> None:
        self.latch(reason)
        raise SafetyViolation(reason)

    def _validate_state(self, state: RobotState) -> None:
        if len(state.positions) != len(self.robot.joint_names):
            self._fail(f"state has {len(state.positions)} positions; expected 14")
        if not all(math.isfinite(value) for value in state.positions):
            self._fail("state contains non-finite positions")
        age = self.clock() - state.captured_at
        if age < -0.01 or age > self.safety.max_state_age_s:
            self._fail(f"state age {age:.3f}s is outside the accepted window")

    def validate_target(
        self,
        state: RobotState,
        target: Sequence[float],
        dt: float,
    ) -> tuple[float, ...]:
        if self._latched_reason is not None:
            raise SafetyViolation(f"safety is latched: {self._latched_reason}")
        self._validate_state(state)
        values = tuple(float(value) for value in target)
        if len(values) != len(self.robot.joint_names):
            self._fail(f"action has {len(values)} positions; expected 14")
        if not math.isfinite(dt) or dt <= 0:
            self._fail("action interval must be positive and finite")
        if not all(math.isfinite(value) for value in values):
            self._fail("action contains non-finite positions")
        for index, (current, goal, low, high, velocity) in enumerate(
            zip(
                state.positions,
                values,
                self.robot.lower_limits,
                self.robot.upper_limits,
                self.robot.max_velocity,
            )
        ):
            if goal < low or goal > high:
                self._fail(f"joint {self.robot.joint_names[index]} target is outside calibrated limits")
            maximum_delta = velocity * dt
            if abs(goal - current) > maximum_delta + 1e-9:
                self._fail(
                    f"joint {self.robot.joint_names[index]} exceeds calibrated velocity "
                    f"({abs(goal-current):.6f} > {maximum_delta:.6f})"
                )
        self._validate_trajectory(state.positions, values)
        return values

    def _validate_trajectory(
        self, current: tuple[float, ...], target: tuple[float, ...]
    ) -> None:
        if self.trajectory_validator is None:
            if self.robot.motion_enabled:
                self._fail("no calibrated trajectory validator is configured")
            return
        movement_time = max(
            abs(goal - start) / velocity
            for start, goal, velocity in zip(current, target, self.robot.max_velocity)
        )
        steps = max(1, math.ceil(movement_time / self.safety.interpolation_step_s))
        for step in range(1, steps + 1):
            ratio = step / steps
            sample = tuple(start + ratio * (goal - start) for start, goal in zip(current, target))
            try:
                accepted = self.trajectory_validator(sample)
            except Exception as error:
                self._fail(f"trajectory validator failed: {error}")
            if accepted is not True:
                self._fail(f"trajectory rejected at interpolation sample {step}/{steps}")
