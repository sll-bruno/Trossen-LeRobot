import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trossen_experiment.adapters import AdapterError, DreamerAdapter, JointNormalizer
from trossen_experiment.cameras import SyntheticCameraHub
from trossen_experiment.dataset import RawEpisodeWriter, validate_episode
from trossen_experiment.evaluation import (
    EvaluationTrial,
    alternating_trials,
    append_result,
    summarize_results,
    write_manifest,
)
from trossen_experiment.models import CameraFrame, EpisodeContext, Observation, Outcome, RobotState
from trossen_experiment.robot import SafeRobot, SimulatedBackend
from trossen_experiment.safety import SafetyValidator
from trossen_experiment.session import SessionRunner, finalize_session
from trossen_experiment.sources import LeaderSource, ProcessPolicySource

from tests.helpers import Clock, make_config


def observation(contacts=None):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    frame = CameraFrame("physical_high", image, 1.0, 1.0)
    front = CameraFrame("physical_front", image, 1.0, 1.0)
    return Observation(
        RobotState.from_values((0.0,) * 14, 1.0),
        {"physical_high": frame, "physical_front": front},
        (0.0,) * 14,
        1.0,
        contacts,
        "validated_sensor" if contacts is not None else None,
        "v1" if contacts is not None else None,
    )


class AdaptersEvaluationSessionTests(unittest.TestCase):
    def test_joint_normalization_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            normalizer = JointNormalizer(make_config(Path(directory)).robot)
            self.assertTrue(np.allclose(normalizer.normalize((0.0,) * 14), 0.0))
            self.assertTrue(np.allclose(normalizer.denormalize((0.0,) * 14), 0.0))

    def test_dreamer_refuses_missing_contacts_and_primes_history(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = DreamerAdapter(
                make_config(Path(directory)).robot,
                stack_obs=3,
                image_transform=lambda name, image: image,
                camera_mapping={"frame_cam_high": "physical_high", "frame_cam_front": "physical_front"},
            )
            with self.assertRaisesRegex(AdapterError, "four pad-contact"):
                adapter.append(observation())
            history = adapter.append(observation((True, False, False, False)))
            self.assertEqual(len(history), 3)
            self.assertEqual(history[-1]["collisions"].shape, (4,))

    def test_evaluation_schedule_and_summary_keep_abort_denominator_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            methods = [("dreamer", "d.pt", 0), ("act5", "a.pt", 0)]
            trials = alternating_trials(methods, config.positions, repetitions=2)
            self.assertEqual(len(trials), 40)
            self.assertEqual({trial.split for trial in trials}, {"train", "test"})
            results = root / "results.jsonl"
            append_result(results, trials[0], Outcome.SUCCESS, "one", 1.0, "height/contact")
            second = EvaluationTrial("dreamer", "d.pt", 0, "train", "train_1", 1)
            append_result(results, second, Outcome.ABORTED, "two", 0.2, "camera stale")
            summary = summarize_results(results)[0]
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(summary["determinate_trials"], 1)
            self.assertEqual(summary["all_attempts"], 2)
            self.assertEqual(summary["aborted"], 1)
            self.assertEqual(summary["task_success_rate"], 1.0)
            self.assertEqual(summary["operational_success_rate"], 0.5)
            self.assertEqual(summary["by_position"]["train_1"]["all_attempts"], 2)
            with self.assertRaisesRegex(ValueError, "already exists"):
                append_result(results, second, Outcome.FAILURE, "three", 1.0, "duplicate")

    def test_three_method_schedule_rotates_first_position(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            methods = [("dreamer", "d", 0), ("act5", "a5", 0), ("act25", "a25", 0)]
            trials = alternating_trials(methods, config.positions, repetitions=3)
            blocks = [trials[index : index + 3] for index in range(0, len(trials), 3)]
            first_counts = {name: 0 for name, _, _ in methods}
            for block in blocks:
                self.assertEqual({trial.method for trial in block}, set(first_counts))
                first_counts[block[0].method] += 1
            self.assertLessEqual(max(first_counts.values()) - min(first_counts.values()), 1)

    def test_frozen_manifest_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = [EvaluationTrial("dreamer", "d", 0, "train", "train_1", 0)]
            write_manifest(path, first)
            write_manifest(path, first)
            changed = [EvaluationTrial("act", "a", 0, "train", "train_1", 0)]
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_manifest(path, changed)

    def test_session_records_monotonic_observation_action_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock(0.0)
            config = make_config(root)
            backend = SimulatedBackend(clock=clock)
            robot = SafeRobot(
                backend, SafetyValidator(config.robot, config.safety, clock=clock), clock=clock
            )
            robot.connect()
            cameras = SyntheticCameraHub(config.cameras, clock=clock)
            writer = RawEpisodeWriter.create(root, "leader", "lift ball", "train_1", "train", True)
            runner = SessionRunner(robot, cameras, 10.0, clock=clock, sleep=clock.sleep)
            result = runner.collect(LeaderSource(robot), writer, 0.15)
            finalize_session(writer, result)
            self.assertGreaterEqual(result.frames, 2)
            self.assertEqual(validate_episode(writer.directory), [])
            rows = [
                json.loads(line)
                for line in (writer.directory / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["frame_index"] for row in rows], list(range(len(rows))))
            self.assertTrue(all(row["action_sent"] == row["action_requested"] for row in rows))
            metadata = json.loads((writer.directory / "metadata.json").read_text())
            self.assertEqual(metadata["session"]["missed_deadlines"], 0)
            self.assertAlmostEqual(metadata["session"]["duration_s"], result.duration_s)

    def test_source_reset_failure_latches_robot_and_finalizes_abort(self):
        class BrokenSource:
            name = "broken"

            def reset(self, context):
                raise RuntimeError("reset failed")

            def act(self, observation):
                raise AssertionError("act must not run")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock(0.0)
            config = make_config(root)
            backend = SimulatedBackend(clock=clock)
            robot = SafeRobot(
                backend, SafetyValidator(config.robot, config.safety, clock=clock), clock=clock
            )
            robot.connect()
            writer = RawEpisodeWriter.create(root, "broken", "lift ball", "dev_1", "development", True)
            runner = SessionRunner(robot, SyntheticCameraHub(config.cameras, clock=clock), 10, clock=clock)
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                runner.collect(BrokenSource(), writer, 1.0)
            self.assertEqual(writer.directory.suffix, ".episode")
            metadata = json.loads((writer.directory / "metadata.json").read_text())
            self.assertEqual(metadata["outcome"], "aborted")
            self.assertIsNotNone(robot.safety.latched_reason)

    def test_partial_send_is_preserved_in_aborted_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock(0.0)
            config = make_config(root)
            backend = SimulatedBackend(clock=clock, fail_side="right")
            backend.leader_positions = (0.01,) * 14
            robot = SafeRobot(
                backend, SafetyValidator(config.robot, config.safety, clock=clock), clock=clock
            )
            robot.connect()
            writer = RawEpisodeWriter.create(root, "leader", "lift ball", "train_1", "train", True)
            runner = SessionRunner(robot, SyntheticCameraHub(config.cameras, clock=clock), 10, clock=clock)
            with self.assertRaises(Exception):
                runner.collect(LeaderSource(robot), writer, 1.0)
            rows = [json.loads(line) for line in (writer.directory / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["partial_sides"], ["left"])
            self.assertEqual(rows[-1]["action_sent"][:7], [0.01] * 7)
            self.assertEqual(rows[-1]["action_sent"][7:], [0.0] * 7)

    def test_policy_inference_runs_in_a_separate_process(self):
        source = ProcessPolicySource(
            "constant",
            "tests.helpers:constant_policy_factory",
            {"value": 0.125},
            timeout_s=2.0,
        )
        context = EpisodeContext("episode", "constant", "task", "dev_1", "development", True)
        try:
            source.prepare()
            source.reset(context)
            self.assertEqual(source.act(observation()), (0.125,) * 14)
        finally:
            source.close()


if __name__ == "__main__":
    unittest.main()
