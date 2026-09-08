"""Command-line entry points for preflight, collection, export, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from PIL import Image, ImageDraw

from .cameras import CameraProcess, SyntheticCameraHub
from .config import ExperimentConfig, import_callable, load_config
from .dataset import (
    RawEpisodeWriter,
    annotate_episode,
    export_lerobot,
    successful_training_episodes,
    validate_episode,
    write_subset_manifests,
)
from .evaluation import EvaluationTrial, alternating_trials, append_result, summarize_results, write_manifest
from .models import Outcome
from .robot import LeRobotBackend, SafeRobot, SimulatedBackend
from .safety import SafetyValidator
from .session import SessionRunner, finalize_session
from .sources import LeaderSource, ProcessPolicySource


class ConsoleOutcomeSignal:
    """Reads one operator annotation without blocking the control loop."""

    VALUES = {
        "s": (Outcome.SUCCESS, "operator confirmed height and contact"),
        "f": (Outcome.FAILURE, "operator marked task failure"),
        "i": (Outcome.INDETERMINATE, "contact or height could not be confirmed"),
        "a": (Outcome.ABORTED, "operator aborted episode"),
    }

    def __init__(self) -> None:
        self.value = None
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        print("During the episode, type s/f/i/a + Enter (success/failure/indeterminate/abort).")
        self._thread.start()

    def _read(self) -> None:
        while self.value is None:
            line = sys.stdin.readline()
            if not line:
                return
            candidate = self.VALUES.get(line.strip().lower())
            if candidate is not None:
                self.value = candidate

    def __call__(self, observation):
        return self.value


def _revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_robot(config: ExperimentConfig, backend_name: str):
    trajectory = (
        None
        if config.safety.trajectory_validator is None
        else import_callable(config.safety.trajectory_validator)
    )
    safety = SafetyValidator(config.robot, config.safety, trajectory_validator=trajectory)
    if backend_name == "simulated":
        backend = SimulatedBackend()
    else:
        if not config.robot.motion_enabled:
            raise RuntimeError("physical backend is disabled in the configuration")
        factory = import_callable(config.robot.factory or "")
        backend = LeRobotBackend(factory)
    return SafeRobot(backend, safety)


def _build_cameras(config: ExperimentConfig, backend_name: str):
    if backend_name == "simulated":
        return SyntheticCameraHub(config.cameras)
    return CameraProcess(config.cameras)


def _metadata(config_path: Path, config: ExperimentConfig) -> dict:
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "repository_revision": _revision(Path.cwd()),
        "control_hz": config.control_hz,
        "calibration_id": config.robot.calibration_id,
    }


def command_preflight(args) -> int:
    config = load_config(args.config)
    print(f"configuration valid: schema={config.schema_version}, cameras={len(config.cameras)}")
    if args.backend == "simulated":
        robot = _build_robot(config, "simulated")
        cameras = _build_cameras(config, "simulated")
        robot.connect()
        cameras.start()
        try:
            state = robot.read_state()
            frames = cameras.latest()
            print(f"simulated state={len(state.positions)} positions, frames={sorted(frames)}")
        finally:
            cameras.close()
            robot.close()
    elif args.cameras:
        cameras = CameraProcess(config.cameras)
        cameras.start()
        end = time.monotonic() + args.duration
        count = 0
        last_frames = None
        try:
            while time.monotonic() < end:
                last_frames = cameras.latest()
                count += 1
        finally:
            cameras.close()
        print(f"camera preflight passed: {count} complete bundles")
        if args.preview_dir:
            preview_dir = Path(args.preview_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in (last_frames or {}).items():
                image = Image.fromarray(frame.image, mode="RGB")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 300, 30), fill="black")
                draw.text((8, 8), name, fill="yellow")
                image.save(preview_dir / f"{name}.png")
            print(f"labeled identity previews written to {preview_dir}")
    else:
        print("physical config validated without connecting cameras or robot")
    return 0


def _collect(args, mode: str) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    config.validate_position(args.position, args.split)
    source = None
    contact_provider = None
    source_kwargs = None
    if mode == "rollout":
        source_kwargs = json.loads(args.source_kwargs)
        source = ProcessPolicySource(
            args.source_name,
            args.source_factory,
            source_kwargs,
            timeout_s=args.inference_timeout,
            startup_timeout_s=args.policy_startup_timeout,
        )
        # Loading a checkpoint may take minutes; it happens before cameras and arms.
        try:
            source.prepare()
        except Exception:
            source.close()
            raise
        if args.contact_factory:
            contact_provider = import_callable(args.contact_factory)(
                **json.loads(args.contact_kwargs)
            )
    try:
        cameras = _build_cameras(config, args.backend)
    except Exception:
        if source is not None:
            source.close()
        raise
    cameras_started = False
    try:
        if args.backend == "physical":
            # Camera health is established before importing/instantiating the arm factory.
            cameras.start()
            cameras.latest(timeout_s=2.0)
            cameras_started = True
        robot = _build_robot(config, args.backend)
        robot.connect()
    except Exception:
        if source is not None:
            source.close()
        cameras.close()
        raise
    if source is None:
        source = LeaderSource(robot)
    try:
        metadata = _metadata(config_path, config)
        if mode == "rollout":
            metadata.update(
                {
                    "source_factory": args.source_factory,
                    "source_kwargs": source_kwargs,
                    "contact_factory": args.contact_factory,
                    "contact_kwargs": (
                        json.loads(args.contact_kwargs) if args.contact_factory else None
                    ),
                }
            )
        writer = RawEpisodeWriter.create(
            config.output_dir,
            source=source.name,
            task=config.task.name,
            position_id=args.position,
            split=args.split,
            pilot=args.pilot,
            metadata=metadata,
        )
        runner = SessionRunner(
            robot, cameras, config.control_hz, contact_provider=contact_provider
        )
        outcome_signal = None
        if mode in {"record", "rollout"} and args.outcome is None and sys.stdin.isatty():
            outcome_signal = ConsoleOutcomeSignal()
            outcome_signal.start()
        result = runner.collect(
            source,
            writer,
            args.duration,
            outcome_signal=outcome_signal,
            cameras_started=cameras_started,
        )
        if mode in {"record", "rollout"} and args.outcome is not None:
            outcome = Outcome(args.outcome)
            reason = args.reason or result.reason
            result = type(result)(
                outcome,
                reason,
                result.frames,
                result.duration_s,
                result.missed_deadlines,
                result.outcome_at_monotonic_s,
            )
            if outcome == Outcome.ABORTED:
                robot.stop(reason)
        costs = {}
        for item in args.cost:
            name, seconds = item.split("=", 1)
            costs[name] = float(seconds)
        finalize_session(writer, result, costs_s=costs)
        print(json.dumps({**result.__dict__, "outcome": result.outcome.value, "episode": str(writer.directory)}))
    finally:
        cameras.close()
        robot.close()
    return 0


def command_validate(args) -> int:
    root = Path(args.path)
    episodes = [root] if root.suffix in {".episode", ".partial"} else sorted(root.glob("*.*"))
    failed = False
    for episode in episodes:
        if episode.suffix not in {".episode", ".partial"}:
            continue
        errors = validate_episode(episode)
        print(json.dumps({"episode": str(episode), "valid": not errors, "errors": errors}))
        failed = failed or bool(errors)
    return 1 if failed else 0


def command_export(args) -> int:
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        episodes = [Path(value) for value in manifest["episodes"]]
    else:
        episodes = successful_training_episodes(Path(args.raw))
    export_lerobot(episodes, Path(args.output), args.repo_id, args.fps, use_videos=not args.images)
    print(f"exported {len(episodes)} episodes to {args.output}")
    return 0


def command_subsets(args) -> int:
    config = load_config(args.config)
    written = write_subset_manifests(
        Path(args.raw), Path(args.output), args.budget, config.positions["train"]
    )
    print(json.dumps({budget: str(path) for budget, path in written.items()}, indent=2))
    return 0


def command_annotate(args) -> int:
    annotate_episode(
        Path(args.episode),
        Outcome(args.outcome),
        args.reason,
        args.reviewer,
        success_frame_index=args.success_frame,
        review_seconds=args.review_seconds,
    )
    print(f"approved review for {args.episode}")
    return 0


def command_manifest(args) -> int:
    config = load_config(args.config)
    methods = []
    for value in args.method:
        parts = value.split(",")
        if len(parts) not in {2, 3}:
            raise ValueError("--method must be NAME,CHECKPOINT[,SEED]")
        if not parts[0] or not parts[1]:
            raise ValueError("method name and checkpoint must be non-empty")
        methods.append((parts[0], parts[1], None if len(parts) == 2 else int(parts[2])))
    if len(methods) != len(set(methods)):
        raise ValueError("evaluation methods must be unique")
    trials = alternating_trials(methods, config.positions, args.repetitions)
    write_manifest(Path(args.output), trials)
    print(f"wrote {len(trials)} fixed trials to {args.output}")
    return 0


def command_summarize(args) -> int:
    print(json.dumps(summarize_results(Path(args.results)), indent=2, sort_keys=True, allow_nan=False))
    return 0


def command_record_result(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.trial_index < 0:
        raise ValueError("trial index must be non-negative")
    try:
        value = manifest[args.trial_index]
    except IndexError as error:
        raise ValueError(f"trial index {args.trial_index} is outside the manifest") from error
    trial = EvaluationTrial(**value)
    episode_path = Path(args.episode)
    errors = validate_episode(episode_path)
    if errors:
        raise ValueError(f"evaluation episode is invalid: {errors}")
    metadata = json.loads((episode_path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("outcome") != args.outcome:
        raise ValueError("recorded outcome does not match reviewed episode metadata")
    if metadata.get("review", {}).get("status") != "approved":
        raise ValueError("evaluation episode has not been approved by video/data review")
    episode_info = metadata.get("episode", {})
    if episode_info.get("split") != trial.split or episode_info.get("position_id") != trial.position_id:
        raise ValueError("episode split/position does not match the frozen evaluation trial")
    recorded_duration = metadata.get("session", {}).get("duration_s")
    if recorded_duration is None:
        raise ValueError("evaluation episode does not contain a recorded session duration")
    if args.duration is not None and abs(args.duration - recorded_duration) > 1e-6:
        raise ValueError("provided duration does not match reviewed episode metadata")
    append_result(
        Path(args.results),
        trial,
        Outcome(args.outcome),
        str(episode_path.resolve()),
        recorded_duration,
        args.reason,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trossen-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--backend", choices=("simulated", "physical"), default="simulated")
    preflight.add_argument("--cameras", action="store_true", help="capture physical cameras without connecting arms")
    preflight.add_argument("--duration", type=float, default=600.0)
    preflight.add_argument("--preview-dir")
    preflight.set_defaults(function=command_preflight)

    for name in ("teleop", "record", "rollout"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--backend", choices=("simulated", "physical"), default="simulated")
        command.add_argument("--duration", type=float, required=True)
        command.add_argument("--position", required=True)
        command.add_argument("--split", choices=("train", "development", "test"), required=True)
        command.add_argument("--pilot", action="store_true")
        command.add_argument(
            "--cost", action="append", default=[], metavar="NAME=SECONDS",
            help="record demonstration/reset/supervision/configuration time",
        )
        if name in {"record", "rollout"}:
            command.add_argument("--outcome", choices=tuple(item.value for item in Outcome))
            command.add_argument("--reason")
        if name == "rollout":
            command.add_argument("--source-name", required=True)
            command.add_argument("--source-factory", required=True)
            command.add_argument("--source-kwargs", default="{}", help="JSON passed to the policy factory")
            command.add_argument("--inference-timeout", type=float, default=0.1)
            command.add_argument("--policy-startup-timeout", type=float, default=120.0)
            command.add_argument("--contact-factory")
            command.add_argument("--contact-kwargs", default="{}")
        command.set_defaults(function=lambda args, mode=name: _collect(args, mode))

    validate = subparsers.add_parser("validate-dataset")
    validate.add_argument("path")
    validate.set_defaults(function=command_validate)

    annotate = subparsers.add_parser("annotate-episode")
    annotate.add_argument("episode")
    annotate.add_argument("--outcome", choices=tuple(item.value for item in Outcome), required=True)
    annotate.add_argument("--reason", required=True)
    annotate.add_argument("--reviewer", required=True)
    annotate.add_argument("--success-frame", type=int)
    annotate.add_argument("--review-seconds", type=float)
    annotate.set_defaults(function=command_annotate)

    export = subparsers.add_parser("export-lerobot")
    export.add_argument("--raw", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--fps", type=int, required=True)
    export.add_argument("--manifest", help="subset manifest created by make-subsets")
    export.add_argument("--images", action="store_true", help="store images instead of encoding videos")
    export.set_defaults(function=command_export)

    subsets = subparsers.add_parser("make-subsets")
    subsets.add_argument("--raw", required=True)
    subsets.add_argument("--config", required=True)
    subsets.add_argument("--output", required=True)
    subsets.add_argument("--budget", action="append", type=int, required=True)
    subsets.set_defaults(function=command_subsets)

    manifest = subparsers.add_parser("evaluate")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--method", action="append", required=True)
    manifest.add_argument("--repetitions", type=int, default=2)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=command_manifest)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("results")
    summarize.set_defaults(function=command_summarize)

    result = subparsers.add_parser("record-result")
    result.add_argument("--results", required=True)
    result.add_argument("--manifest", required=True)
    result.add_argument("--trial-index", type=int, required=True)
    result.add_argument("--outcome", choices=tuple(item.value for item in Outcome), required=True)
    result.add_argument("--episode", required=True)
    result.add_argument(
        "--duration",
        type=float,
        help="optional cross-check; the recorded episode duration remains authoritative",
    )
    result.add_argument("--reason", required=True)
    result.set_defaults(function=command_record_result)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
