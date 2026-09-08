"""Policy adapters that make preprocessing and unit conversions explicit."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from .config import RobotConfig
from .models import Observation


class AdapterError(RuntimeError):
    pass


class JointNormalizer:
    def __init__(self, config: RobotConfig) -> None:
        self.low = np.asarray(config.lower_limits, dtype=np.float32)
        self.high = np.asarray(config.upper_limits, dtype=np.float32)

    def normalize(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (14,):
            raise AdapterError(f"expected 14 positions, got {array.shape}")
        return 2.0 * (array - self.low) / (self.high - self.low) - 1.0

    def denormalize(self, values: Sequence[float]) -> tuple[float, ...]:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (14,):
            raise AdapterError(f"expected 14 normalized actions, got {array.shape}")
        if not np.isfinite(array).all() or (np.abs(array) > 1.0 + 1e-6).any():
            raise AdapterError("normalized action must be finite and inside [-1, 1]")
        physical = (array + 1.0) * 0.5 * (self.high - self.low) + self.low
        return tuple(float(value) for value in physical)


class DreamerAdapter:
    """Builds the existing two-camera, joint/target, and contact contract.

    Resizing, cropping, normalization statistics, and camera geometry are
    checkpoint-specific and must be supplied by ``image_transform``. The
    adapter deliberately has no guessed fallback.
    """

    def __init__(
        self,
        robot: RobotConfig,
        stack_obs: int,
        image_transform,
        camera_mapping: Mapping[str, str],
    ) -> None:
        if stack_obs <= 0:
            raise AdapterError("stack_obs must be positive")
        if set(camera_mapping) != {"frame_cam_high", "frame_cam_front"}:
            raise AdapterError("Dreamer camera mapping must define frame_cam_high and frame_cam_front")
        self.normalizer = JointNormalizer(robot)
        self.stack_obs = stack_obs
        self.image_transform = image_transform
        self.camera_mapping = dict(camera_mapping)
        self.history: deque[dict[str, Any]] = deque(maxlen=stack_obs)

    def reset(self) -> None:
        self.history.clear()

    def append(self, observation: Observation) -> list[dict[str, Any]]:
        if observation.contacts is None or len(observation.contacts) != 4:
            raise AdapterError("Dreamer requires four pad-contact values")
        if not observation.contact_source or not observation.contact_adapter_version:
            raise AdapterError("Dreamer contacts require source and adapter version")
        if observation.previous_target is None:
            target = observation.state.positions
        else:
            target = observation.previous_target
        converted: dict[str, Any] = {
            "joint_position": self.normalizer.normalize(observation.state.positions),
            "target_position": self.normalizer.normalize(target),
            "collisions": np.asarray(observation.contacts, dtype=np.bool_),
        }
        for policy_name, physical_name in self.camera_mapping.items():
            try:
                image = observation.frames[physical_name].image
            except KeyError as error:
                raise AdapterError(f"missing physical camera {physical_name!r}") from error
            converted[policy_name] = self.image_transform(policy_name, np.asarray(image))
        if not self.history:
            for _ in range(self.stack_obs - 1):
                self.history.append({key: np.copy(value) for key, value in converted.items()})
        self.history.append(converted)
        return list(self.history)


class ACTPolicyAdapter:
    """Converts physical observations for a loaded LeRobot ACTPolicy."""

    def __init__(self, policy: Any, device: str, camera_mapping: Mapping[str, str]) -> None:
        try:
            import torch
        except ImportError as error:
            raise AdapterError("ACT inference must run in the pinned LeRobot environment") from error
        self.torch = torch
        self.policy = policy
        self.device = torch.device(device)
        self.camera_mapping = dict(camera_mapping)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, observation: Observation) -> tuple[float, ...]:
        torch = self.torch
        batch = {
            "observation.state": torch.as_tensor(
                observation.state.positions, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        }
        for policy_name, physical_name in self.camera_mapping.items():
            if physical_name not in observation.frames:
                raise AdapterError(f"missing physical camera {physical_name!r}")
            image = torch.as_tensor(
                np.asarray(observation.frames[physical_name].image), device=self.device
            )
            batch[policy_name] = image.to(torch.float32).div(255).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy.select_action(batch).squeeze(0).detach().cpu().numpy()
        if action.shape != (14,) or not np.isfinite(action).all():
            raise AdapterError(f"ACT returned an invalid action with shape {action.shape}")
        return tuple(float(value) for value in action)
