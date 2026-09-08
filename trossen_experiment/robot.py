"""Robot ownership and adapters for simulated and LeRobot-backed execution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol, Sequence

from .models import ActionResult, RobotState
from .safety import SafetyValidator, SafetyViolation


class RobotError(RuntimeError):
    pass


class PartialSendError(RobotError):
    def __init__(self, sent_sides: tuple[str, ...], cause: Exception) -> None:
        self.sent_sides = sent_sides
        self.requested_target: tuple[float, ...] | None = None
        self.sent_target: tuple[float, ...] | None = None
        self.sent_at: float | None = None
        super().__init__(f"partial bimanual send ({sent_sides}): {cause}")


class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def read_state(self) -> RobotState: ...
    def read_leaders(self) -> tuple[float, ...]: ...
    def send_targets(self, target: Sequence[float]) -> tuple[float, ...]: ...
    def stop(self, reason: str) -> None: ...
    def close(self) -> None: ...


class SafeRobot:
    """The only route from an action source to a backend."""

    def __init__(
        self,
        backend: RobotBackend,
        safety: SafetyValidator,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.safety = safety
        self.clock = clock
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            raise RobotError("robot is already connected")
        self.backend.connect()
        self.connected = True

    def read_state(self) -> RobotState:
        if not self.connected:
            raise RobotError("robot is not connected")
        return self.backend.read_state()

    def read_leaders(self) -> tuple[float, ...]:
        if not self.connected:
            raise RobotError("robot is not connected")
        return self.backend.read_leaders()

    def send(self, state: RobotState, requested: Sequence[float], dt: float) -> ActionResult:
        if not self.connected:
            raise RobotError("robot is not connected")
        validated = self.safety.validate_target(state, requested, dt)
        try:
            sent = self.backend.send_targets(validated)
        except PartialSendError as error:
            combined = list(state.positions)
            for side in error.sent_sides:
                offset = 0 if side == "left" else 7
                combined[offset : offset + 7] = validated[offset : offset + 7]
            error.requested_target = validated
            error.sent_target = tuple(combined)
            error.sent_at = self.clock()
            self.safety.latch(str(error))
            self.backend.stop(str(error))
            raise
        except Exception as error:
            self.safety.latch(f"send failed: {error}")
            self.backend.stop(str(error))
            raise RobotError(f"send failed: {error}") from error
        return ActionResult(validated, tuple(sent), self.clock())

    def stop(self, reason: str) -> None:
        self.safety.latch(reason)
        self.backend.stop(reason)

    def clear_after_operator_reset(self) -> RobotState:
        resync = getattr(self.backend, "operator_resync", None)
        if callable(resync):
            resync()
        state = self.read_state()
        self.safety.clear(state)
        return state

    def close(self) -> None:
        if self.connected:
            abort_close = getattr(self.backend, "abort_close", None)
            if self.safety.latched_reason is not None and callable(abort_close):
                abort_close()
            else:
                self.backend.close()
            self.connected = False


@dataclass
class SimulatedBackend:
    """Deterministic backend used by development, CI, and CLI smoke tests."""

    positions: tuple[float, ...] = (0.0,) * 14
    leader_positions: tuple[float, ...] = (0.0,) * 14
    clock: Callable[[], float] = time.monotonic
    fail_side: str | None = None
    connected: bool = False
    stopped_reason: str | None = None

    def connect(self) -> None:
        self.connected = True

    def read_state(self) -> RobotState:
        if not self.connected:
            raise RobotError("simulated backend is disconnected")
        return RobotState.from_values(self.positions, self.clock(), (0.0,) * 14)

    def read_leaders(self) -> tuple[float, ...]:
        return self.leader_positions

    def send_targets(self, target: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in target)
        sent: list[str] = []
        for side, section in (("left", values[:7]), ("right", values[7:])):
            if self.fail_side == side:
                raise PartialSendError(tuple(sent), RuntimeError(f"simulated {side} failure"))
            if side == "left":
                self.positions = tuple(section) + self.positions[7:]
            else:
                self.positions = self.positions[:7] + tuple(section)
            sent.append(side)
        return values

    def stop(self, reason: str) -> None:
        self.stopped_reason = reason

    def close(self) -> None:
        self.connected = False


class LeRobotBackend:
    """Thin adapter around an already configured LeRobot ManipulatorRobot.

    The supplied factory owns version-specific configuration. This adapter does
    not guess IP addresses, camera identities, limits, or calibration.
    """

    def __init__(self, robot_factory: Callable[[], object], clock: Callable[[], float] = time.monotonic):
        self.robot_factory = robot_factory
        self.clock = clock
        self.robot: object | None = None
        self._software_halted = False

    def connect(self) -> None:
        if self.robot is not None:
            raise RobotError("LeRobot backend is already connected")
        robot = self.robot_factory()
        required = (
            "abort_connection",
            "experiment_stop",
            "experiment_close",
            "experiment_abort_close",
            "experiment_resync",
        )
        missing = [name for name in required if not callable(getattr(robot, name, None))]
        if missing:
            raise RobotError(
                "physical robot factory lacks calibrated lifecycle methods: " + ", ".join(missing)
            )
        try:
            robot.connect()
        except Exception:
            abort = getattr(robot, "abort_connection", None)
            if callable(abort):
                abort()
            raise
        self.robot = robot
        self._software_halted = False

    def _arms(self, name: str):
        if self.robot is None:
            raise RobotError("LeRobot backend is disconnected")
        arms = getattr(self.robot, name)
        if tuple(arms) != ("left", "right"):
            raise RobotError(f"{name} must be ordered exactly as left, right")
        return arms

    def read_state(self) -> RobotState:
        captured_at = self.clock()
        positions: list[float] = []
        efforts: list[float] = []
        for arm in self._arms("follower_arms").values():
            positions.extend(float(value) for value in arm.read("Present_Position"))
            efforts.extend(float(value) for value in arm.read("External_Efforts"))
        return RobotState.from_values(positions, captured_at, efforts)

    def read_leaders(self) -> tuple[float, ...]:
        positions: list[float] = []
        for arm in self._arms("leader_arms").values():
            positions.extend(float(value) for value in arm.read("Present_Position"))
        return tuple(positions)

    def send_targets(self, target: Sequence[float]) -> tuple[float, ...]:
        if self._software_halted:
            raise RobotError("backend is software-halted; operator reset is required")
        values = tuple(float(value) for value in target)
        sent_sides: list[str] = []
        for index, (side, arm) in enumerate(self._arms("follower_arms").items()):
            try:
                arm.write("Goal_Position", values[index * 7 : (index + 1) * 7])
            except Exception as error:
                self._software_halted = True
                raise PartialSendError(tuple(sent_sides), error) from error
            sent_sides.append(side)
        return values

    def stop(self, reason: str) -> None:
        self._software_halted = True
        if self.robot is not None:
            self.robot.experiment_stop(reason)

    def operator_resync(self) -> None:
        if self.robot is None:
            raise RobotError("LeRobot backend is disconnected")
        self.robot.experiment_resync()
        self.read_state()
        self._software_halted = False

    def close(self) -> None:
        if self.robot is None:
            return
        self.robot.experiment_close()
        self.robot = None
        self._software_halted = True

    def abort_close(self) -> None:
        if self.robot is None:
            return
        self.robot.experiment_abort_close()
        self.robot = None
        self._software_halted = True
