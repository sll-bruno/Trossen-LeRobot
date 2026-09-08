"""One deadline-driven loop for teleoperation, collection, and policies."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Protocol

from .dataset import RawEpisodeWriter
from .models import ActionResult, Observation, Outcome
from .robot import PartialSendError, SafeRobot
from .sources import ActionSource


class CameraHub(Protocol):
    def start(self) -> None: ...
    def latest(self, timeout_s: float = 1.0): ...
    def close(self, timeout_s: float = 2.0) -> None: ...


ContactProvider = Callable[[Observation], tuple[tuple[bool, ...], str, str] | None]
OutcomeSignal = Callable[[Observation], tuple[Outcome, str] | None]


@dataclass(frozen=True)
class SessionResult:
    outcome: Outcome
    reason: str
    frames: int
    duration_s: float
    missed_deadlines: int
    outcome_at_monotonic_s: float | None = None


class SessionRunner:
    def __init__(
        self,
        robot: SafeRobot,
        cameras: CameraHub,
        control_hz: float,
        contact_provider: ContactProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.cameras = cameras
        self.period_s = 1.0 / control_hz
        self.contact_provider = contact_provider
        self.clock = clock
        self.sleep = sleep

    def collect(
        self,
        source: ActionSource,
        writer: RawEpisodeWriter,
        horizon_s: float,
        outcome_signal: OutcomeSignal | None = None,
        cameras_started: bool = False,
    ) -> SessionResult:
        if horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        start = self.clock()
        deadline = start
        previous_target: tuple[float, ...] | None = None
        previous_observation: Observation | None = None
        frames = 0
        missed = 0
        try:
            source.reset(writer.context)
            if not cameras_started:
                self.cameras.start()
            while True:
                now = self.clock()
                if outcome_signal is not None and previous_observation is not None:
                    outcome = outcome_signal(previous_observation)
                    if outcome is not None:
                        if outcome[0] == Outcome.ABORTED:
                            self.robot.stop(outcome[1])
                        return SessionResult(outcome[0], outcome[1], frames, now - start, missed, now)
                if now - start >= horizon_s:
                    return SessionResult(Outcome.INDETERMINATE, "horizon reached", frames, now - start, missed)
                if now < deadline:
                    self.sleep(deadline - now)
                    now = self.clock()
                elif now - deadline > self.period_s:
                    missed += int((now - deadline) // self.period_s)
                    deadline = now

                state = self.robot.read_state()
                camera_frames = self.cameras.latest(timeout_s=max(1.0, self.period_s * 2))
                observation = Observation(state, camera_frames, previous_target, self.clock())
                if self.contact_provider is not None:
                    contact = self.contact_provider(observation)
                    if contact is not None:
                        values, contact_source, version = contact
                        observation = Observation(
                            state,
                            camera_frames,
                            previous_target,
                            observation.observed_at,
                            tuple(bool(value) for value in values),
                            contact_source,
                            version,
                        )
                requested = source.act(observation)
                try:
                    action = self.robot.send(state, requested, self.period_s)
                except PartialSendError as error:
                    if (
                        error.requested_target is not None
                        and error.sent_target is not None
                        and error.sent_at is not None
                    ):
                        writer.append(
                            observation,
                            ActionResult(
                                error.requested_target,
                                error.sent_target,
                                error.sent_at,
                                error.sent_sides,
                            ),
                        )
                        frames += 1
                    raise
                writer.append(observation, action)
                frames += 1
                previous_target = action.sent
                previous_observation = observation
                if outcome_signal is not None:
                    outcome = outcome_signal(observation)
                    if outcome is not None:
                        completed_at = self.clock()
                        if outcome[0] == Outcome.ABORTED:
                            self.robot.stop(outcome[1])
                        return SessionResult(
                            outcome[0], outcome[1], frames, completed_at - start, missed, completed_at
                        )
                deadline += self.period_s
        except BaseException as error:
            try:
                self.robot.stop(f"episode aborted: {error}")
            finally:
                writer.abort(
                    str(error),
                    duration_s=max(0.0, self.clock() - start),
                    missed_deadlines=missed,
                )
            raise
        finally:
            close_source = getattr(source, "close", None)
            if callable(close_source):
                close_source()
            self.cameras.close()


def finalize_session(
    writer: RawEpisodeWriter,
    result: SessionResult,
    costs_s: Mapping[str, float] | None = None,
) -> None:
    writer.finalize(
        result.outcome,
        result.reason,
        costs_s=costs_s,
        duration_s=result.duration_s,
        missed_deadlines=result.missed_deadlines,
        success_at_monotonic_s=(
            result.outcome_at_monotonic_s if result.outcome == Outcome.SUCCESS else None
        ),
    )
