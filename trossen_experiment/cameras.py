"""Camera capture in an isolated process with freshness checks."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import time
import traceback
from typing import Any, Callable, Mapping, Protocol

from .config import CameraConfig, import_callable
from .models import CameraFrame


class CameraError(RuntimeError):
    pass


class CameraDriver(Protocol):
    def open(self) -> None: ...
    def read(self) -> Any: ...
    def close(self) -> None: ...


class OpenCVCameraDriver:
    """OpenCV driver for an explicitly mapped device; imports cv2 only in the worker."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise CameraError("opencv-python is required for the OpenCV camera driver") from error
        capture = cv2.VideoCapture(self.config.device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"cannot open camera {self.config.name!r} at {self.config.device!r}")
        self.capture = capture

    def read(self):
        if self.capture is None:
            raise CameraError("camera is not open")
        ok, bgr = self.capture.read()
        if not ok:
            raise CameraError(f"camera {self.config.name!r} returned no frame")
        return bgr[:, :, ::-1].copy()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def open_cv_driver(config: CameraConfig) -> OpenCVCameraDriver:
    return OpenCVCameraDriver(config)


class RealSenseCameraDriver:
    """Color-only D405 capture selected by persistent serial number."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.pipeline = None

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise CameraError("pyrealsense2 is required for the RealSense camera driver") from error
        pipeline = rs.pipeline()
        settings = rs.config()
        settings.enable_device(str(self.config.device))
        settings.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            int(self.config.fps),
        )
        pipeline.start(settings)
        self.pipeline = pipeline

    def read(self):
        if self.pipeline is None:
            raise CameraError("camera is not open")
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        if not color:
            raise CameraError(f"camera {self.config.name!r} returned no color frame")
        try:
            import numpy as np

            return np.asanyarray(color.get_data()).copy()
        except Exception as error:
            raise CameraError(f"cannot copy RGB frame from {self.config.name!r}: {error}") from error

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


def realsense_driver(config: CameraConfig) -> RealSenseCameraDriver:
    return RealSenseCameraDriver(config)


def _put_latest(queue: mp.Queue, value: Any) -> None:
    try:
        queue.put_nowait(value)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        return
    try:
        queue.put_nowait(value)
    except Full:
        # multiprocessing.Queue uses a feeder thread; a concurrent put may win
        # between get and put. Dropping this bundle is safer than killing capture.
        return


def _camera_worker(
    configs: tuple[CameraConfig, ...],
    frame_queue: mp.Queue,
    error_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    drivers: dict[str, CameraDriver] = {}
    previous_images: dict[str, Any] = {}
    identical_counts = {config.name: 0 for config in configs}
    try:
        for config in configs:
            factory = import_callable(config.driver)
            driver = factory(config)
            driver.open()
            drivers[config.name] = driver
        while not stop_event.is_set():
            images: dict[str, tuple[float, Any]] = {}
            for config in configs:
                captured_at = time.monotonic()
                image = drivers[config.name].read()
                if config.name in previous_images and _images_equal(previous_images[config.name], image):
                    identical_counts[config.name] += 1
                else:
                    identical_counts[config.name] = 0
                if identical_counts[config.name] >= config.max_identical_frames:
                    raise CameraError(
                        f"camera {config.name!r} returned {identical_counts[config.name] + 1} identical frames"
                    )
                previous_images[config.name] = image.copy()
                images[config.name] = (captured_at, image)
            _put_latest(frame_queue, (time.monotonic(), images))
    except BaseException:
        _put_latest(error_queue, traceback.format_exc())
    finally:
        for driver in drivers.values():
            try:
                driver.close()
            except Exception:
                pass


def _images_equal(first: Any, second: Any) -> bool:
    try:
        import numpy as np

        return np.array_equal(first, second)
    except Exception:
        return first == second


class CameraProcess:
    def __init__(self, configs: tuple[CameraConfig, ...], context: str = "spawn") -> None:
        self.configs = configs
        self._context = mp.get_context(context)
        self._frames = self._context.Queue(maxsize=1)
        self._errors = self._context.Queue(maxsize=1)
        self._stop = self._context.Event()
        self._process: mp.Process | None = None

    def start(self) -> None:
        if self._process is not None:
            raise CameraError("camera process is already started")
        self._stop.clear()
        self._process = self._context.Process(
            target=_camera_worker,
            args=(self.configs, self._frames, self._errors, self._stop),
            daemon=True,
        )
        self._process.start()

    def latest(self, timeout_s: float = 1.0) -> Mapping[str, CameraFrame]:
        self._raise_worker_error()
        try:
            received_at, values = self._frames.get(timeout=timeout_s)
        except Empty as error:
            self._raise_worker_error()
            raise CameraError("timed out waiting for camera frames") from error
        now = time.monotonic()
        output = {}
        config_by_name = {config.name: config for config in self.configs}
        for name, (captured_at, image) in values.items():
            age = now - captured_at
            if age < -0.01 or age > config_by_name[name].max_age_s:
                raise CameraError(f"camera {name!r} frame is stale ({age:.3f}s)")
            shape = getattr(image, "shape", None)
            expected = (config_by_name[name].height, config_by_name[name].width, 3)
            if shape != expected:
                raise CameraError(f"camera {name!r} shape is {shape}; expected {expected}")
            output[name] = CameraFrame(name, image, captured_at, received_at)
        if set(output) != set(config_by_name):
            raise CameraError("camera bundle is incomplete")
        return output

    def _raise_worker_error(self) -> None:
        try:
            message = self._errors.get_nowait()
        except Empty:
            return
        raise CameraError(f"camera worker failed:\n{message}")

    def close(self, timeout_s: float = 2.0) -> None:
        process = self._process
        if process is None:
            return
        self._stop.set()
        process.join(timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(timeout_s)
        self._process = None


class StaticCameraHub:
    """In-process camera double used in tests and simulation."""

    def __init__(self, frames: Mapping[str, CameraFrame]) -> None:
        self.frames = frames

    def start(self) -> None:
        return None

    def latest(self, timeout_s: float = 1.0) -> Mapping[str, CameraFrame]:
        return self.frames

    def close(self, timeout_s: float = 2.0) -> None:
        return None


class SyntheticCameraHub:
    """Fresh black images for end-to-end smoke tests without camera hardware."""

    def __init__(self, configs: tuple[CameraConfig, ...], clock: Callable[[], float] = time.monotonic) -> None:
        self.configs = configs
        self.clock = clock

    def start(self) -> None:
        return None

    def latest(self, timeout_s: float = 1.0) -> Mapping[str, CameraFrame]:
        import numpy as np

        now = self.clock()
        return {
            config.name: CameraFrame(
                config.name,
                np.zeros((config.height, config.width, 3), dtype=np.uint8),
                now,
                now,
            )
            for config in self.configs
        }

    def close(self, timeout_s: float = 2.0) -> None:
        return None
