"""Small immutable values shared by the experiment modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORTED = "aborted"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class RobotState:
    positions: tuple[float, ...]
    captured_at: float
    external_efforts: tuple[float, ...] | None = None

    @classmethod
    def from_values(
        cls,
        positions: Sequence[float],
        captured_at: float,
        external_efforts: Sequence[float] | None = None,
    ) -> "RobotState":
        return cls(
            tuple(float(value) for value in positions),
            float(captured_at),
            None if external_efforts is None else tuple(float(value) for value in external_efforts),
        )


@dataclass(frozen=True)
class CameraFrame:
    name: str
    image: Any
    captured_at: float
    received_at: float


@dataclass(frozen=True)
class Observation:
    state: RobotState
    frames: Mapping[str, CameraFrame]
    previous_target: tuple[float, ...] | None
    observed_at: float
    contacts: tuple[bool, ...] | None = None
    contact_source: str | None = None
    contact_adapter_version: str | None = None


@dataclass(frozen=True)
class ActionResult:
    requested: tuple[float, ...]
    sent: tuple[float, ...]
    sent_at: float
    partial_sides: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    source: str
    task: str
    position_id: str
    split: str
    pilot: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
