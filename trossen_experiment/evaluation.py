"""Fixed evaluation manifests and transparent result summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from .models import Outcome


@dataclass(frozen=True)
class EvaluationTrial:
    method: str
    checkpoint: str
    seed: int | None
    split: str
    position_id: str
    repetition: int


def alternating_trials(
    methods: list[tuple[str, str, int | None]],
    positions: dict[str, tuple[str, ...]],
    repetitions: int = 2,
) -> list[EvaluationTrial]:
    """Build a deterministic, position-balanced schedule across reset blocks."""
    if not methods:
        raise ValueError("at least one evaluation method is required")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    trials: list[EvaluationTrial] = []
    block = 0
    for split in ("train", "test"):
        for position in positions[split]:
            for repetition in range(repetitions):
                offset = block % len(methods)
                ordered = methods[offset:] + methods[:offset]
                if (block // len(methods)) % 2:
                    ordered = list(reversed(ordered))
                for method, checkpoint, seed in ordered:
                    trials.append(EvaluationTrial(method, checkpoint, seed, split, position, repetition))
                block += 1
    return trials


def write_manifest(path: Path, trials: Iterable[EvaluationTrial]) -> None:
    values = [trial.__dict__ for trial in trials]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(values, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise ValueError(f"refusing to overwrite frozen evaluation manifest {path}")
    path.write_text(content, encoding="utf-8")


def append_result(
    path: Path,
    trial: EvaluationTrial,
    outcome: Outcome,
    episode_id: str,
    duration_s: float,
    reason: str,
) -> None:
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("result duration must be finite and non-negative")
    if not episode_id or not reason:
        raise ValueError("result episode and reason are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **trial.__dict__,
        "outcome": outcome.value,
        "episode_id": episode_id,
        "duration_s": duration_s,
        "reason": reason,
    }
    if path.exists():
        trial_fields = ("method", "checkpoint", "seed", "split", "position_id", "repetition")
        with path.open(encoding="utf-8") as existing:
            for line in existing:
                previous = json.loads(line)
                if all(previous.get(field) == row.get(field) for field in trial_fields):
                    raise ValueError(f"result already exists for trial {trial}")
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, separators=(",", ":")) + "\n")


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total <= 0:
        return (None, None)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def summarize_results(path: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("outcome") not in {outcome.value for outcome in Outcome}:
                raise ValueError(f"unknown evaluation outcome {row.get('outcome')!r}")
            key = (row["method"], row["checkpoint"], row.get("seed"), row["split"])
            groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        metrics = _metrics(rows)
        by_position = {
            position: _metrics([row for row in rows if row["position_id"] == position])
            for position in sorted({row["position_id"] for row in rows})
        }
        summaries.append(
            {
                "method": key[0],
                "checkpoint": key[1],
                "seed": key[2],
                "split": key[3],
                **metrics,
                "by_position": by_position,
            }
        )
    return summaries


def _metrics(rows: list[dict]) -> dict:
    successes = sum(row["outcome"] == Outcome.SUCCESS.value for row in rows)
    determinate = sum(
        row["outcome"] in {Outcome.SUCCESS.value, Outcome.FAILURE.value} for row in rows
    )
    attempts = len(rows)
    low, high = wilson_interval(successes, determinate)
    return {
        "successes": successes,
        "determinate_trials": determinate,
        "all_attempts": attempts,
        "aborted": sum(row["outcome"] == Outcome.ABORTED.value for row in rows),
        "indeterminate": sum(row["outcome"] == Outcome.INDETERMINATE.value for row in rows),
        "task_success_rate": None if determinate == 0 else successes / determinate,
        "operational_success_rate": None if attempts == 0 else successes / attempts,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }
