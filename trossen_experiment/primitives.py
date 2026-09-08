"""Optional calibrated waypoint primitive for later automatic collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import Observation


class WaypointPrimitive:
    """Interpolates explicit joint targets; safety remains in ``SafeRobot``."""

    def __init__(self, waypoints: list[Mapping[str, Any]]) -> None:
        if not waypoints:
            raise ValueError("primitive requires at least one waypoint")
        self.waypoints = []
        for index, waypoint in enumerate(waypoints):
            target = tuple(float(value) for value in waypoint["target"])
            steps = int(waypoint["steps"])
            if len(target) != 14 or steps <= 0:
                raise ValueError(f"waypoint {index} requires 14 targets and positive steps")
            self.waypoints.append((target, steps))
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.step = 0
        self.start = None

    def __call__(self, observation: Observation) -> tuple[float, ...]:
        if self.index >= len(self.waypoints):
            return self.waypoints[-1][0]
        target, steps = self.waypoints[self.index]
        if self.start is None:
            self.start = observation.state.positions
        self.step += 1
        ratio = min(1.0, self.step / steps)
        action = tuple(
            initial + ratio * (goal - initial) for initial, goal in zip(self.start, target)
        )
        if self.step >= steps:
            self.index += 1
            self.step = 0
            self.start = target
        return action


def waypoint_policy_factory(path: str) -> WaypointPrimitive:
    """Load a versioned primitive JSON for ``ProcessPolicySource``."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported primitive schema_version")
    return WaypointPrimitive(value["waypoints"])
