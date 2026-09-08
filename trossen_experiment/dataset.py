"""Crash-recoverable raw episodes and optional LeRobot export."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import Full, Queue
import threading
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
from PIL import Image

from .models import ActionResult, EpisodeContext, Observation, Outcome


class DatasetError(RuntimeError):
    pass


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_finite_vector(value: Any, size: int) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape == (size,) and bool(np.isfinite(array).all())


class RawEpisodeWriter:
    """Writes every transition before an episode is accepted for training."""

    def __init__(self, root: Path, context: EpisodeContext, started_at: float | None = None) -> None:
        self.root = Path(root)
        self.context = context
        self.started_at = time.time() if started_at is None else started_at
        self.directory = self.root / f"{context.episode_id}.partial"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.images_dir = self.directory / "images"
        self.images_dir.mkdir()
        self.frames_path = self.directory / "frames.jsonl"
        self._frames = self.frames_path.open("a", encoding="utf-8")
        self.frame_count = 0
        self._finalized = False
        self._write_queue: Queue = Queue(maxsize=8)
        self._writer_error: BaseException | None = None
        _atomic_json(
            self.directory / "metadata.json",
            {
                "schema_version": 1,
                "episode": asdict(context),
                "started_at_unix_s": self.started_at,
                "status": "recording",
            },
        )
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _write_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                row, images = item
                if self._writer_error is not None:
                    continue
                for relative, image in images.items():
                    destination = self.directory / relative
                    temporary = destination.with_suffix(".png.tmp")
                    Image.fromarray(image, mode="RGB").save(temporary, format="PNG")
                    os.replace(temporary, destination)
                self._frames.write(
                    json.dumps(row, separators=(",", ":"), default=_json_default) + "\n"
                )
                self._frames.flush()
                os.fsync(self._frames.fileno())
            except BaseException as error:
                self._writer_error = error
            finally:
                self._write_queue.task_done()

    def _finish_writes(self) -> None:
        self._write_queue.put(None)
        self._write_queue.join()
        self._writer_thread.join()
        self._frames.close()
        if self._writer_error is not None:
            raise DatasetError(f"background episode writer failed: {self._writer_error}")

    @classmethod
    def create(cls, root: Path, source: str, task: str, position_id: str, split: str, pilot: bool,
               metadata: Mapping[str, Any] | None = None) -> "RawEpisodeWriter":
        context = EpisodeContext(
            episode_id=uuid.uuid4().hex,
            source=source,
            task=task,
            position_id=position_id,
            split=split,
            pilot=pilot,
            metadata={} if metadata is None else dict(metadata),
        )
        return cls(root, context)

    def append(self, observation: Observation, action: ActionResult) -> None:
        if self._finalized:
            raise DatasetError("episode is already finalized")
        if self._writer_error is not None:
            raise DatasetError(f"background episode writer failed: {self._writer_error}")
        image_paths: dict[str, str] = {}
        images: dict[str, np.ndarray] = {}
        for name, frame in observation.frames.items():
            relative = Path("images") / f"{self.frame_count:06d}_{name}.png"
            image = np.asarray(frame.image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise DatasetError(f"camera {name!r} must provide an HxWx3 uint8 RGB image")
            images[str(relative)] = image.copy()
            image_paths[name] = str(relative)
        row = {
            "frame_index": self.frame_count,
            "observed_at_monotonic_s": observation.observed_at,
            "state_captured_at_monotonic_s": observation.state.captured_at,
            "positions": observation.state.positions,
            "external_efforts": observation.state.external_efforts,
            "previous_target": observation.previous_target,
            "contacts": observation.contacts,
            "contact_source": observation.contact_source,
            "contact_adapter_version": observation.contact_adapter_version,
            "camera_captured_at_monotonic_s": {
                name: frame.captured_at for name, frame in observation.frames.items()
            },
            "camera_received_at_monotonic_s": {
                name: frame.received_at for name, frame in observation.frames.items()
            },
            "images": image_paths,
            "action_requested": action.requested,
            "action_sent": action.sent,
            "action_sent_at_monotonic_s": action.sent_at,
            "partial_sides": action.partial_sides,
        }
        try:
            self._write_queue.put_nowait((row, images))
        except Full as error:
            raise DatasetError("episode writer queue is full") from error
        self.frame_count += 1

    def finalize(
        self,
        outcome: Outcome,
        reason: str,
        costs_s: Mapping[str, float] | None = None,
        success_at_monotonic_s: float | None = None,
        duration_s: float | None = None,
        missed_deadlines: int | None = None,
    ) -> Path:
        if self._finalized:
            raise DatasetError("episode is already finalized")
        self._finish_writes()
        _atomic_json(
            self.directory / "metadata.json",
            {
                "schema_version": 1,
                "episode": asdict(self.context),
                "started_at_unix_s": self.started_at,
                "finished_at_unix_s": time.time(),
                "status": "complete",
                "outcome": outcome.value,
                "reason": reason,
                "frame_count": self.frame_count,
                "success_at_monotonic_s": success_at_monotonic_s,
                "costs_s": {} if costs_s is None else dict(costs_s),
                "session": {
                    "duration_s": duration_s,
                    "missed_deadlines": missed_deadlines,
                },
                "review": {"status": "pending"},
            },
        )
        destination = self.root / f"{self.context.episode_id}.episode"
        os.replace(self.directory, destination)
        self.directory = destination
        self._finalized = True
        return destination

    def abort(
        self,
        reason: str,
        duration_s: float | None = None,
        missed_deadlines: int | None = None,
    ) -> Path:
        return self.finalize(
            Outcome.ABORTED,
            reason,
            duration_s=duration_s,
            missed_deadlines=missed_deadlines,
        )

    def __enter__(self) -> "RawEpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._finalized:
            self.abort("context exited" if exc is None else f"exception: {exc}")


def iter_rows(episode: Path) -> Iterable[dict[str, Any]]:
    with (episode / "frames.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{episode}: invalid frame JSON at line {line_number}") from error


def annotate_episode(
    episode: Path,
    outcome: Outcome,
    reason: str,
    reviewer: str,
    success_frame_index: int | None = None,
    review_seconds: float | None = None,
) -> None:
    episode = Path(episode)
    errors = validate_episode(episode)
    if errors:
        raise DatasetError(f"cannot approve invalid episode {episode}: {errors}")
    if not reviewer.strip():
        raise DatasetError("reviewer must be recorded")
    rows = list(iter_rows(episode))
    success_at = None
    if outcome == Outcome.SUCCESS:
        if success_frame_index is None:
            raise DatasetError("success requires the reviewed success frame index")
        if success_frame_index < 0 or success_frame_index >= len(rows):
            raise DatasetError("success frame index is outside the episode")
        success_at = rows[success_frame_index]["observed_at_monotonic_s"]
    elif success_frame_index is not None:
        raise DatasetError("success frame index is only valid for a success")
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("outcome") == Outcome.ABORTED.value and outcome != Outcome.ABORTED:
        raise DatasetError("a technical abort cannot be relabelled as a task result")
    if outcome == Outcome.SUCCESS and any(row.get("partial_sides") for row in rows):
        raise DatasetError("an episode with a partial bimanual send cannot be approved as success")
    metadata.update(
        {
            "outcome": outcome.value,
            "reason": reason,
            "success_at_monotonic_s": success_at,
            "review": {
                "status": "approved",
                "reviewer": reviewer,
                "reviewed_at_unix_s": time.time(),
            },
        }
    )
    if review_seconds is not None:
        if review_seconds < 0:
            raise DatasetError("review_seconds cannot be negative")
        metadata.setdefault("costs_s", {})["review"] = float(review_seconds)
    _atomic_json(metadata_path, metadata)


def validate_episode(episode: Path) -> list[str]:
    errors: list[str] = []
    metadata_path = episode / "metadata.json"
    if not metadata_path.exists():
        return ["metadata.json is missing"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = list(iter_rows(episode)) if (episode / "frames.jsonl").exists() else []
    if episode.suffix == ".partial":
        errors.append("episode is incomplete")
    if episode.suffix == ".episode" and metadata.get("status") != "complete":
        errors.append("final episode metadata is not complete")
    if metadata.get("frame_count", len(rows)) != len(rows):
        errors.append("metadata frame_count does not match frames.jsonl")
    previous_time: float | None = None
    camera_shapes: dict[str, tuple[int, ...]] = {}
    for index, row in enumerate(rows):
        if row.get("frame_index") != index:
            errors.append(f"frame {index}: non-contiguous frame_index")
        timestamp = row.get("observed_at_monotonic_s")
        if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
            errors.append(f"frame {index}: observation timestamp is not finite")
            continue
        if previous_time is not None and timestamp <= previous_time:
            errors.append(f"frame {index}: observation timestamps are not increasing")
        previous_time = timestamp
        image_names = set(row.get("images", {}))
        if image_names != {"cam_high", "cam_front"}:
            errors.append(f"frame {index}: images must contain exactly cam_high and cam_front")
        captured = row.get("camera_captured_at_monotonic_s", {})
        received = row.get("camera_received_at_monotonic_s", {})
        if set(captured) != image_names or set(received) != image_names:
            errors.append(f"frame {index}: camera timestamp keys do not match images")
        for name, relative in row.get("images", {}).items():
            path = episode / relative
            if not path.exists():
                errors.append(f"frame {index}: missing image {name!r}")
                continue
            try:
                image = np.asarray(Image.open(path))
                if image.ndim != 3 or image.shape[-1] != 3:
                    errors.append(f"frame {index}: image {name!r} is not HxWx3")
                elif name in camera_shapes and image.shape != camera_shapes[name]:
                    errors.append(f"frame {index}: camera {name!r} shape changed within episode")
                else:
                    camera_shapes[name] = image.shape
            except Exception as error:
                errors.append(f"frame {index}: unreadable image {name!r}: {error}")
        for field in ("positions", "action_requested", "action_sent"):
            values = row.get(field, [])
            if not _is_finite_vector(values, 14):
                errors.append(f"frame {index}: {field} must contain 14 finite values")
        previous_target = row.get("previous_target")
        if previous_target is not None and not _is_finite_vector(previous_target, 14):
            errors.append(f"frame {index}: previous_target must be null or 14 finite values")
        efforts = row.get("external_efforts")
        if efforts is not None and not _is_finite_vector(efforts, 14):
            errors.append(f"frame {index}: external_efforts must be null or 14 finite values")
        contacts = row.get("contacts")
        if contacts is not None and (
            len(contacts) != 4
            or not row.get("contact_source")
            or not row.get("contact_adapter_version")
        ):
            errors.append(f"frame {index}: contacts lack four values, source, or adapter version")
    return errors


def successful_training_episodes(root: Path) -> list[Path]:
    selected = []
    for episode in sorted(Path(root).glob("*.episode")):
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        info = metadata.get("episode", {})
        if (
            metadata.get("outcome") == Outcome.SUCCESS.value
            and metadata.get("review", {}).get("status") == "approved"
            and info.get("split") == "train"
            and not info.get("pilot", False)
            and not validate_episode(episode)
        ):
            selected.append(episode)
    return selected


def balanced_nested_subsets(
    root: Path,
    budgets: Iterable[int],
    expected_positions: Iterable[str] | None = None,
) -> dict[int, list[Path]]:
    """Select deterministic nested subsets, cycling once through positions first."""
    episodes = successful_training_episodes(root)
    by_position: dict[str, list[Path]] = {}
    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        position = metadata["episode"]["position_id"]
        by_position.setdefault(position, []).append(episode)
    if expected_positions is not None:
        expected = set(expected_positions)
        found = set(by_position)
        if found != expected:
            raise DatasetError(
                f"training successes cover positions {sorted(found)}; expected exactly {sorted(expected)}"
            )
    ordered: list[Path] = []
    depth = 0
    while True:
        added = False
        for position in sorted(by_position):
            if depth < len(by_position[position]):
                ordered.append(by_position[position][depth])
                added = True
        if not added:
            break
        depth += 1
    result = {}
    for budget in sorted(set(int(value) for value in budgets)):
        if budget <= 0:
            raise DatasetError("budgets must be positive")
        if len(ordered) < budget:
            raise DatasetError(f"budget {budget} requires {budget} successes; only {len(ordered)} are valid")
        result[budget] = ordered[:budget]
    return result


def write_subset_manifests(
    root: Path,
    output: Path,
    budgets: Iterable[int],
    expected_positions: Iterable[str] | None = None,
) -> dict[int, Path]:
    subsets = balanced_nested_subsets(root, budgets, expected_positions)
    output.mkdir(parents=True, exist_ok=True)
    written = {}
    for budget, episodes in subsets.items():
        path = output / f"act_{budget}.json"
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "budget": budget,
                "selection": "deterministic_nested_position_round_robin",
                "episodes": [str(episode.resolve()) for episode in episodes],
            },
        )
        written[budget] = path
    return written


def export_lerobot(
    episodes: Iterable[Path],
    output: Path,
    repo_id: str,
    fps: int,
    use_videos: bool = True,
) -> None:
    """Export reviewed raw successes through the installed LeRobotDataset API."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise DatasetError("run export inside the pinned LeRobot environment") from error
    paths = list(episodes)
    if not paths:
        raise DatasetError("no reviewed training successes were selected")
    for episode in paths:
        errors = validate_episode(episode)
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        info = metadata.get("episode", {})
        if errors:
            raise DatasetError(f"episode {episode} is invalid: {errors}")
        if metadata.get("outcome") != Outcome.SUCCESS.value:
            raise DatasetError(f"episode {episode} is not a reviewed success")
        if metadata.get("review", {}).get("status") != "approved":
            raise DatasetError(f"episode {episode} has not been approved by video/data review")
        if info.get("split") != "train" or info.get("pilot", False):
            raise DatasetError(f"episode {episode} is not a non-pilot training episode")
    first_row = next(iter(iter_rows(paths[0])))
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {"dtype": "float32", "shape": (14,), "names": None},
    }
    for name, relative in first_row["images"].items():
        shape = tuple(np.asarray(Image.open(paths[0] / relative)).shape)
        features[f"observation.images.{name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output,
        robot_type="trossen_ai_stationary",
        features=features,
        use_videos=use_videos,
    )
    try:
        for episode in paths:
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            task = metadata["episode"]["task"]
            for row in iter_rows(episode):
                frame = {
                    "observation.state": np.asarray(row["positions"], dtype=np.float32),
                    "action": np.asarray(row["action_sent"], dtype=np.float32),
                    "task": task,
                }
                for name, relative in row["images"].items():
                    frame[f"observation.images.{name}"] = np.asarray(Image.open(episode / relative))
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        stop_writer = getattr(dataset, "stop_image_writer", None)
        if callable(stop_writer):
            stop_writer()
