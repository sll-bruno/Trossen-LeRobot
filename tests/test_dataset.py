import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trossen_experiment.dataset import (
    RawEpisodeWriter,
    annotate_episode,
    balanced_nested_subsets,
    export_lerobot,
    successful_training_episodes,
    validate_episode,
)
from trossen_experiment.models import ActionResult, CameraFrame, Observation, Outcome, RobotState


def append_frame(writer, timestamp=1.0, value=0.0):
    state = RobotState.from_values((value,) * 14, timestamp)
    pixels = np.full((64, 64, 3), round(value * 1000) % 256, dtype=np.uint8)
    frames = {
        name: CameraFrame(name, pixels, timestamp, timestamp)
        for name in ("cam_high", "cam_front")
    }
    observation = Observation(state, frames, None, timestamp)
    action = ActionResult((value,) * 14, (value,) * 14, timestamp)
    writer.append(observation, action)


def make_success(root: Path, position: str) -> Path:
    writer = RawEpisodeWriter.create(root, "leader", "lift ball", position, "train", False)
    append_frame(writer)
    episode = writer.finalize(Outcome.SUCCESS, "provisional")
    annotate_episode(episode, Outcome.SUCCESS, "reviewed", "test", success_frame_index=0)
    return episode


class DatasetTests(unittest.TestCase):
    def test_episode_is_atomic_and_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RawEpisodeWriter.create(root, "leader", "lift ball", "train_1", "train", True)
            append_frame(writer)
            self.assertEqual(writer.directory.suffix, ".partial")
            destination = writer.finalize(Outcome.INDETERMINATE, "contact not confirmed")
            self.assertEqual(destination.suffix, ".episode")
            self.assertEqual(validate_episode(destination), [])
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertEqual(metadata["outcome"], "indeterminate")

    def test_only_reviewed_nonpilot_train_success_is_exportable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = make_success(root, "train_1")
            pilot = RawEpisodeWriter.create(root, "leader", "lift ball", "train_1", "train", True)
            append_frame(pilot)
            pilot.finalize(Outcome.SUCCESS, "pilot")
            failed = RawEpisodeWriter.create(root, "leader", "lift ball", "train_1", "train", False)
            append_frame(failed)
            failed.finalize(Outcome.FAILURE, "failed")
            self.assertEqual(successful_training_episodes(root), [accepted])

    def test_nested_subsets_are_balanced_by_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for _depth in range(2):
                for position in ("train_1", "train_2", "train_3", "train_4", "train_5"):
                    make_success(root, position)
            subsets = balanced_nested_subsets(root, (5, 10))
            self.assertTrue(set(subsets[5]).issubset(subsets[10]))
            first_positions = {
                json.loads((episode / "metadata.json").read_text())["episode"]["position_id"]
                for episode in subsets[5]
            }
            self.assertEqual(len(first_positions), 5)

    def test_exported_dataset_runs_one_real_act_optimization_step(self):
        try:
            import torch
            from torch.utils.data import DataLoader
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
            from lerobot.common.datasets.utils import dataset_to_policy_features
            from lerobot.common.policies.act.configuration_act import ACTConfig
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.configs.types import FeatureType
        except ImportError:
            self.skipTest("LeRobot is optional outside the robot environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RawEpisodeWriter.create(
                root / "raw", "leader", "lift ball", "train_1", "train", False
            )
            append_frame(writer, 1.0, 0.0)
            append_frame(writer, 1.1, 0.01)
            episode = writer.finalize(Outcome.SUCCESS, "provisional")
            annotate_episode(episode, Outcome.SUCCESS, "reviewed", "test", success_frame_index=0)
            output = root / "lerobot"
            export_lerobot([episode], output, "local/test", 30, use_videos=False)
            info = json.loads((output / "meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 2)

            metadata = LeRobotDatasetMetadata("local/test", root=output)
            features = dataset_to_policy_features(metadata.features)
            inputs = {
                key: feature
                for key, feature in features.items()
                if feature.type in {FeatureType.STATE, FeatureType.VISUAL}
            }
            outputs = {
                key: feature for key, feature in features.items()
                if feature.type is FeatureType.ACTION
            }
            config = ACTConfig(
                input_features=inputs,
                output_features=outputs,
                device="cpu",
                chunk_size=1,
                n_action_steps=1,
                pretrained_backbone_weights=None,
                dim_model=32,
                n_heads=4,
                dim_feedforward=64,
                n_encoder_layers=1,
                n_decoder_layers=1,
                latent_dim=8,
                n_vae_encoder_layers=1,
            )
            policy = ACTPolicy(config, dataset_stats=metadata.stats).train()
            dataset = LeRobotDataset(
                "local/test", root=output, delta_timestamps={"action": [0.0]}
            )
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
            loss, _ = policy.forward(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()

    def test_export_rejects_pilot_even_when_manifest_is_hand_edited(self):
        try:
            import lerobot  # noqa: F401
        except ImportError:
            self.skipTest("LeRobot is optional outside the robot environment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RawEpisodeWriter.create(
                root / "raw", "leader", "lift ball", "train_1", "train", True
            )
            append_frame(writer)
            episode = writer.finalize(Outcome.SUCCESS, "pilot")
            annotate_episode(episode, Outcome.SUCCESS, "reviewed pilot", "test", success_frame_index=0)
            with self.assertRaisesRegex(Exception, "non-pilot"):
                export_lerobot([episode], root / "lerobot", "local/test", 30, use_videos=False)

    def test_technical_abort_cannot_be_approved_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RawEpisodeWriter.create(
                root, "leader", "lift ball", "train_1", "train", False
            )
            append_frame(writer)
            episode = writer.finalize(Outcome.ABORTED, "partial send")
            with self.assertRaisesRegex(Exception, "technical abort"):
                annotate_episode(
                    episode,
                    Outcome.SUCCESS,
                    "looked successful",
                    "test",
                    success_frame_index=0,
                )


if __name__ == "__main__":
    unittest.main()
