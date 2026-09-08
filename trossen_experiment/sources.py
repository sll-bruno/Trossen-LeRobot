"""Interchangeable action sources for one shared control loop."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence

from .config import import_callable
from .models import EpisodeContext, Observation
from .robot import SafeRobot


class ActionSource(Protocol):
    name: str
    def reset(self, context: EpisodeContext) -> None: ...
    def act(self, observation: Observation) -> Sequence[float]: ...


class LeaderSource:
    name = "leader"

    def __init__(self, robot: SafeRobot) -> None:
        self.robot = robot

    def reset(self, context: EpisodeContext) -> None:
        return None

    def act(self, observation: Observation) -> tuple[float, ...]:
        return self.robot.read_leaders()


class CallablePolicySource:
    def __init__(self, name: str, policy: Callable[[Observation], Sequence[float]]) -> None:
        self.name = name
        self.policy = policy

    def reset(self, context: EpisodeContext) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Observation) -> tuple[float, ...]:
        return tuple(float(value) for value in self.policy(observation))


class DreamerPolicySource(CallablePolicySource):
    """Preserves the existing Dreamer observation contract."""

    def act(self, observation: Observation) -> tuple[float, ...]:
        if observation.contacts is None or len(observation.contacts) != 4:
            raise RuntimeError("Dreamer requires four validated pad-contact channels")
        if not observation.contact_source or not observation.contact_adapter_version:
            raise RuntimeError("Dreamer contacts require a documented source and adapter version")
        return super().act(observation)


def _policy_worker(factory_path: str, factory_kwargs: dict, requests, responses) -> None:
    try:
        policy = import_callable(factory_path)(**factory_kwargs)
        while True:
            request_id, command, payload = requests.get()
            if command == "close":
                return
            if command == "prepare":
                responses.put((request_id, True, None))
                continue
            if command == "reset":
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                responses.put((request_id, True, None))
                continue
            action = tuple(float(value) for value in policy(payload))
            responses.put((request_id, True, action))
    except BaseException:
        responses.put((-1, False, traceback.format_exc()))


class ProcessPolicySource:
    """Runs inference outside the process that owns the robot backend."""

    def __init__(
        self,
        name: str,
        factory_path: str,
        factory_kwargs: Mapping[str, Any],
        timeout_s: float,
        startup_timeout_s: float = 120.0,
        context: str = "spawn",
    ) -> None:
        self.name = name
        self.factory_path = factory_path
        self.factory_kwargs = dict(factory_kwargs)
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self._context = mp.get_context(context)
        self._requests = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = None
        self._request_id = 0

    def _start(self) -> None:
        if self._process is not None:
            return
        self._process = self._context.Process(
            target=_policy_worker,
            args=(self.factory_path, self.factory_kwargs, self._requests, self._responses),
            daemon=True,
        )
        self._process.start()

    def _request(self, command: str, payload=None, timeout_s: float | None = None):
        self._start()
        self._request_id += 1
        request_id = self._request_id
        limit = self.timeout_s if timeout_s is None else timeout_s
        try:
            self._requests.put((request_id, command, payload), timeout=min(limit, 0.1))
        except Full as error:
            raise RuntimeError("policy request queue is full") from error
        try:
            response_id, ok, result = self._responses.get(timeout=limit)
        except Empty as error:
            raise RuntimeError(f"policy worker timed out after {limit:.3f}s") from error
        if not ok:
            raise RuntimeError(f"policy worker failed:\n{result}")
        if response_id != request_id:
            raise RuntimeError("policy worker returned a stale response")
        return result

    def reset(self, context: EpisodeContext) -> None:
        self._request("reset", timeout_s=self.startup_timeout_s)

    def prepare(self) -> None:
        """Load the policy before any robot connection is attempted."""
        self._request("prepare", timeout_s=self.startup_timeout_s)

    def act(self, observation: Observation) -> tuple[float, ...]:
        return self._request("act", observation)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            self._requests.put_nowait((self._request_id + 1, "close", None))
        except Full:
            pass
        process.join(1.0)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        self._process = None


def act_policy_factory(checkpoint: str, device: str, camera_mapping: Mapping[str, str]):
    """Factory intended for :class:`ProcessPolicySource`."""
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError as error:
        raise RuntimeError("ACT factory must run in the pinned LeRobot environment") from error
    from .adapters import ACTPolicyAdapter

    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    return ACTPolicyAdapter(policy, device, camera_mapping)
