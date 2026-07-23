from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
except ImportError as exc:  # pragma: no cover - exercised by installations without server extras
    raise RuntimeError(
        "FastAPI server dependencies are unavailable. Install them with "
        "`pip install -e '.[server]'`."
    ) from exc


LOGGER = logging.getLogger(__name__)


class VisionSessionProtocol(Protocol):
    """Small runtime contract used by the WebSocket server and its fakes."""

    model_load_ms: float

    def process_frame(self, frame: np.ndarray, context: object) -> object: ...

    def reset(self) -> None: ...


SessionFactory = Callable[[], VisionSessionProtocol]


def _environment_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _environment_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _environment_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Configuration for one-process, one-live-session server operation."""

    model: str = "yolo26s.pt"
    confidence: float = 0.10
    image_size: int = 640
    device: str | None = None
    classes: tuple[int, ...] | None = None
    tracker: str = "botsort.yaml"
    narrate_bus_approach: bool = False
    kiosk_ocr_interval_frames: int = 5
    text_ocr_interval_frames: int = 5
    max_frame_bytes: int = 4 * 1024 * 1024
    max_frame_width: int = 3840
    max_frame_height: int = 2160
    max_receive_fps: float = 30.0
    max_session_id_length: int = 128
    debug_frame_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.image_size < 1:
            raise ValueError("image_size must be at least 1")
        if not self.tracker.strip():
            raise ValueError("tracker must not be empty")
        for name, value in (
            ("kiosk_ocr_interval_frames", self.kiosk_ocr_interval_frames),
            ("text_ocr_interval_frames", self.text_ocr_interval_frames),
            ("max_frame_bytes", self.max_frame_bytes),
            ("max_frame_width", self.max_frame_width),
            ("max_frame_height", self.max_frame_height),
            ("max_session_id_length", self.max_session_id_length),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not math.isfinite(self.max_receive_fps) or self.max_receive_fps < 0.0:
            raise ValueError("max_receive_fps must be non-negative")
        if self.debug_frame_dir is not None and not isinstance(self.debug_frame_dir, Path):
            object.__setattr__(self, "debug_frame_dir", Path(self.debug_frame_dir))

    @classmethod
    def from_environment(cls) -> ServerConfig:
        """Build the production defaults without adding a settings dependency."""
        device = os.getenv("VISION_SERVER_DEVICE")
        normalized_device = device.strip() if device is not None else ""
        debug_frame_dir = os.getenv("VISION_SERVER_DEBUG_FRAME_DIR")
        normalized_debug_frame_dir = debug_frame_dir.strip() if debug_frame_dir is not None else ""
        return cls(
            model=os.getenv("VISION_SERVER_MODEL", "yolo26s.pt"),
            confidence=_environment_float("VISION_SERVER_CONFIDENCE", 0.10),
            image_size=_environment_int("VISION_SERVER_IMAGE_SIZE", 640),
            device=normalized_device or None,
            tracker=os.getenv("VISION_SERVER_TRACKER", "botsort.yaml"),
            narrate_bus_approach=_environment_bool(
                "VISION_SERVER_NARRATE_BUS_APPROACH",
                False,
            ),
            kiosk_ocr_interval_frames=_environment_int(
                "VISION_SERVER_KIOSK_OCR_INTERVAL_FRAMES",
                5,
            ),
            text_ocr_interval_frames=_environment_int(
                "VISION_SERVER_TEXT_OCR_INTERVAL_FRAMES",
                5,
            ),
            max_frame_bytes=_environment_int(
                "VISION_SERVER_MAX_FRAME_BYTES",
                4 * 1024 * 1024,
            ),
            max_frame_width=_environment_int("VISION_SERVER_MAX_FRAME_WIDTH", 3840),
            max_frame_height=_environment_int("VISION_SERVER_MAX_FRAME_HEIGHT", 2160),
            max_receive_fps=_environment_float("VISION_SERVER_MAX_RECEIVE_FPS", 30.0),
            debug_frame_dir=(
                Path(normalized_debug_frame_dir) if normalized_debug_frame_dir else None
            ),
        )


@dataclass(frozen=True, slots=True)
class _StartMessage:
    session_id: str
    source_width: int
    source_height: int
    source_fps: float


@dataclass(frozen=True, slots=True)
class _FrameHeader:
    sequence_id: int
    captured_at_ms: int | float | None


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    sequence_id: int
    captured_at_ms: int | float | None
    jpeg_bytes: bytes
    server_received_at_ms: int
    received_at_s: float


@dataclass(frozen=True, slots=True)
class _FrameOutcome:
    analysis: object | None
    decode_ms: float
    decode_started_at_s: float
    processing_started_at_s: float
    completed_at_s: float
    error_code: str | None = None
    error_message: str | None = None
    process_invoked: bool = False


@dataclass(slots=True)
class _ConnectionMetrics:
    received_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    rejected_frames: int = 0
    received_at_window: deque[float] | None = None
    completed_at_s: deque[float] | None = None
    total_latency_ms: list[float] | None = None

    def __post_init__(self) -> None:
        self.received_at_window = deque()
        self.completed_at_s = deque(maxlen=30)
        self.total_latency_ms = []

    def processing_fps(self) -> float:
        completed = self.completed_at_s
        if completed is None or len(completed) < 2:
            return 0.0
        elapsed_s = completed[-1] - completed[0]
        return (len(completed) - 1) / elapsed_s if elapsed_s > 0.0 else 0.0


class _SingleSessionGate:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self.active = False

    async def claim(self) -> bool:
        async with self._guard:
            if self.active:
                return False
            self.active = True
            return True

    def release(self) -> None:
        # Endpoint cleanup runs on the owning event loop. Keeping release
        # synchronous prevents task cancellation from stranding the gate.
        self.active = False


def _error_payload(
    code: str,
    message: str,
    *,
    sequence_id: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "error",
        "code": code,
        "message": message,
    }
    if sequence_id is not None:
        payload["sequence_id"] = sequence_id
    return payload


def _parse_json_object(raw_text: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None, "message must be valid JSON"
    if not isinstance(payload, dict):
        return None, "message must be a JSON object"
    return payload, None


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _parse_start_message(
    raw_text: str,
    config: ServerConfig,
) -> tuple[_StartMessage | None, str | None]:
    payload, error = _parse_json_object(raw_text)
    if payload is None:
        return None, error
    if payload.get("type") != "start":
        return None, "the first message must have type 'start'"

    raw_session_id = payload.get("session_id")
    session_id = str(raw_session_id).strip() if isinstance(raw_session_id, str) else ""
    if not session_id:
        return None, "session_id must be a non-empty string"
    if len(session_id) > config.max_session_id_length:
        return None, "session_id is too long"

    width = payload.get("source_width")
    height = payload.get("source_height")
    source_fps = _finite_number(payload.get("source_fps"))
    if not _is_integer(width) or width < 1:
        return None, "source_width must be a positive integer"
    if not _is_integer(height) or height < 1:
        return None, "source_height must be a positive integer"
    if width > config.max_frame_width or height > config.max_frame_height:
        return None, "declared source dimensions exceed the configured maximum"
    if source_fps is None or source_fps <= 0.0:
        return None, "source_fps must be a positive number"
    if config.max_receive_fps > 0.0 and source_fps > config.max_receive_fps:
        return None, "source_fps exceeds the configured maximum"
    return _StartMessage(session_id, width, height, source_fps), None


def _parse_frame_header(raw_text: str) -> tuple[_FrameHeader | None, str | None]:
    payload, error = _parse_json_object(raw_text)
    if payload is None:
        return None, error
    if payload.get("type") != "frame":
        return None, "expected a message with type 'frame'"

    sequence_id = payload.get("sequence_id")
    if not _is_integer(sequence_id) or sequence_id < 0:
        return None, "sequence_id must be a non-negative integer"
    if "captured_at_ms" not in payload:
        return None, "captured_at_ms is required"
    captured_at_ms = payload.get("captured_at_ms")
    if captured_at_ms is not None:
        normalized_capture_time = _finite_number(captured_at_ms)
        if normalized_capture_time is None or normalized_capture_time < 0.0:
            return None, "captured_at_ms must be a non-negative number or null"
    return _FrameHeader(sequence_id, captured_at_ms), None


def _default_session_factory(config: ServerConfig) -> SessionFactory:
    def build_session() -> VisionSessionProtocol:
        # The base package remains usable without importing the optional server or
        # initializing a model. A live model is created only after a valid start.
        from .pipeline import PipelineConfig, create_vision_session

        pipeline_config = PipelineConfig(
            source="<live>",
            model=config.model,
            classes=config.classes,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
            tracker=config.tracker,
            save_crops=False,
            kiosk_ocr_interval_frames=config.kiosk_ocr_interval_frames,
            text_ocr_interval_frames=config.text_ocr_interval_frames,
            narrate_bus_approach=config.narrate_bus_approach,
        )
        return create_vision_session(
            pipeline_config,
            live_mode=True,
            tracker_override=config.tracker,
            narrate_bus_approach=config.narrate_bus_approach,
        )

    return build_session


def _safe_model_load_ms(session: VisionSessionProtocol, fallback_ms: float) -> float:
    try:
        value = float(session.model_load_ms)
    except (AttributeError, TypeError, ValueError):
        return max(0.0, fallback_ms)
    return value if math.isfinite(value) and value >= 0.0 else max(0.0, fallback_ms)


def _process_pending_frame(
    session: VisionSessionProtocol,
    pending: _PendingFrame,
    *,
    processed_index: int,
    dropped_frames: int,
    config: ServerConfig,
) -> _FrameOutcome:
    if config.debug_frame_dir is not None:
        try:
            config.debug_frame_dir.mkdir(parents=True, exist_ok=True)
            debug_path = config.debug_frame_dir / (
                f"frame_{pending.server_received_at_ms}_{pending.sequence_id}.jpg"
            )
            debug_path.write_bytes(pending.jpeg_bytes)
        except OSError:
            LOGGER.warning(
                "debug JPEG save failed for sequence_id=%s",
                pending.sequence_id,
                exc_info=True,
            )
    decode_started_at_s = time.perf_counter()
    encoded = np.frombuffer(pending.jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decode_completed_at_s = time.perf_counter()
    decode_ms = (decode_completed_at_s - decode_started_at_s) * 1000.0
    if frame is None or frame.size == 0:
        return _FrameOutcome(
            analysis=None,
            decode_ms=decode_ms,
            decode_started_at_s=decode_started_at_s,
            processing_started_at_s=decode_completed_at_s,
            completed_at_s=decode_completed_at_s,
            error_code="INVALID_JPEG",
            error_message="binary frame is not a valid JPEG image",
        )

    height, width = frame.shape[:2]
    if width > config.max_frame_width or height > config.max_frame_height:
        return _FrameOutcome(
            analysis=None,
            decode_ms=decode_ms,
            decode_started_at_s=decode_started_at_s,
            processing_started_at_s=decode_completed_at_s,
            completed_at_s=decode_completed_at_s,
            error_code="FRAME_TOO_LARGE",
            error_message="decoded frame dimensions exceed the configured maximum",
        )

    processing_started_at_s = time.perf_counter()
    captured_at_s = (
        float(pending.captured_at_ms) / 1000.0 if pending.captured_at_ms is not None else None
    )
    try:
        from .pipeline import FrameContext

        context = FrameContext(
            source_sequence_id=pending.sequence_id,
            processed_index=processed_index,
            captured_at_s=captured_at_s,
            received_at_s=pending.received_at_s,
            processing_started_at_s=processing_started_at_s,
            dropped_frames=dropped_frames,
        )
        analysis = session.process_frame(frame, context)
    except Exception:
        LOGGER.exception(
            "vision frame processing failed for sequence_id=%s",
            pending.sequence_id,
        )
        try:
            session.reset()
        except Exception:
            LOGGER.exception("vision session reset failed after processing error")
        return _FrameOutcome(
            analysis=None,
            decode_ms=decode_ms,
            decode_started_at_s=decode_started_at_s,
            processing_started_at_s=processing_started_at_s,
            completed_at_s=time.perf_counter(),
            error_code="PROCESSING_FAILED",
            error_message="vision processing failed for this frame",
            process_invoked=True,
        )

    return _FrameOutcome(
        analysis=analysis,
        decode_ms=decode_ms,
        decode_started_at_s=decode_started_at_s,
        processing_started_at_s=processing_started_at_s,
        completed_at_s=time.perf_counter(),
        process_invoked=True,
    )


def _serialized_events(raw_events: object) -> list[dict[str, object]]:
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        return []
    serialized: list[dict[str, object]] = []
    for event in raw_events:
        if isinstance(event, Mapping):
            serialized.append(dict(event))
            continue
        to_dict = getattr(event, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                serialized.append(dict(payload))
    return serialized


def _serialized_narrations(raw_narrations: object) -> list[str]:
    if not isinstance(raw_narrations, Sequence) or isinstance(
        raw_narrations,
        (str, bytes, bytearray),
    ):
        return []
    messages: list[str] = []
    for narration in raw_narrations:
        raw_message = getattr(narration, "message", narration)
        message = str(raw_message).strip()
        if message:
            messages.append(message)
    return messages


def _safe_timings(analysis: object) -> dict[str, float]:
    raw_timings = getattr(analysis, "timings", {})
    if not isinstance(raw_timings, Mapping):
        return {}
    timings: dict[str, float] = {}
    for key, value in raw_timings.items():
        normalized = _finite_number(value)
        if normalized is not None and normalized >= 0.0:
            timings[str(key)] = normalized
    return timings


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def create_app(
    config: ServerConfig | None = None,
    session_factory: SessionFactory | None = None,
) -> FastAPI:
    """Create an injectable single-session FastAPI application."""
    server_config = config or ServerConfig.from_environment()
    build_session = session_factory or _default_session_factory(server_config)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-worker")
    gate = _SingleSessionGate()

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(title="Voice Agent Vision Server", lifespan=lifespan)
    application.state.server_config = server_config
    application.state.session_gate = gate

    @application.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "active_session": gate.active,
        }

    @application.websocket("/ws/vision")
    async def vision_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        if not await gate.claim():
            await websocket.send_json(
                _error_payload(
                    "SESSION_BUSY",
                    "another vision session is already active",
                )
            )
            await websocket.close(code=1013)
            return

        session: VisionSessionProtocol | None = None
        worker_task: asyncio.Task[None] | None = None
        frame_queue: asyncio.Queue[_PendingFrame | None] = asyncio.Queue(maxsize=1)
        send_lock = asyncio.Lock()
        metrics = _ConnectionMetrics()
        session_started_at_s = time.perf_counter()
        model_load_ms = 0.0
        pending_header: _FrameHeader | None = None

        async def safe_send(payload: Mapping[str, object]) -> bool:
            try:
                async with send_lock:
                    await websocket.send_json(dict(payload))
            except (RuntimeError, OSError, WebSocketDisconnect):
                return False
            return True

        async def worker() -> None:
            nonlocal session
            processed_index = 0
            loop = asyncio.get_running_loop()
            while True:
                pending = await frame_queue.get()
                try:
                    if pending is None:
                        return
                    if session is None:
                        return
                    outcome = await loop.run_in_executor(
                        executor,
                        lambda: _process_pending_frame(
                            session,
                            pending,
                            processed_index=processed_index,
                            dropped_frames=metrics.dropped_frames,
                            config=server_config,
                        ),
                    )
                    if outcome.process_invoked:
                        processed_index += 1
                    if outcome.error_code is not None:
                        metrics.rejected_frames += 1
                        await safe_send(
                            _error_payload(
                                outcome.error_code,
                                outcome.error_message or "frame processing failed",
                                sequence_id=pending.sequence_id,
                            )
                        )
                        continue

                    analysis = outcome.analysis
                    if analysis is None:
                        metrics.rejected_frames += 1
                        await safe_send(
                            _error_payload(
                                "PROCESSING_FAILED",
                                "vision processing returned no result",
                                sequence_id=pending.sequence_id,
                            )
                        )
                        continue

                    metrics.processed_frames += 1
                    completed_at_s = outcome.completed_at_s
                    assert metrics.completed_at_s is not None
                    metrics.completed_at_s.append(completed_at_s)
                    total_server_ms = (completed_at_s - pending.received_at_s) * 1000.0
                    assert metrics.total_latency_ms is not None
                    metrics.total_latency_ms.append(total_server_ms)
                    timings = _safe_timings(analysis)
                    timings.update(
                        {
                            "queue_wait_ms": max(
                                0.0,
                                (outcome.decode_started_at_s - pending.received_at_s) * 1000.0,
                            ),
                            "decode_ms": outcome.decode_ms,
                            "inference_ms": timings.get("inference_ms", 0.0),
                            "analysis_ms": timings.get("analysis_ms", 0.0),
                            "total_server_ms": max(0.0, total_server_ms),
                        }
                    )
                    response: dict[str, object] = {
                        "type": "analysis",
                        "sequence_id": pending.sequence_id,
                        "captured_at_ms": pending.captured_at_ms,
                        "server_received_at_ms": pending.server_received_at_ms,
                        "completed_at_ms": time.time_ns() // 1_000_000,
                        "dropped_frames": metrics.dropped_frames,
                        "received_frames": metrics.received_frames,
                        "processed_frames": metrics.processed_frames,
                        "processing_fps": round(metrics.processing_fps(), 3),
                        "model_load_ms": round(model_load_ms, 3),
                        "analysis_events": _serialized_events(
                            getattr(analysis, "analysis_events", ())
                        ),
                        "narrations": _serialized_narrations(getattr(analysis, "narrations", ())),
                        "timings": {key: round(value, 3) for key, value in timings.items()},
                    }
                    await safe_send(response)
                finally:
                    frame_queue.task_done()

        try:
            initial_message = await websocket.receive()
            if initial_message.get("type") == "websocket.disconnect":
                return
            raw_start = initial_message.get("text")
            if not isinstance(raw_start, str):
                await safe_send(
                    _error_payload(
                        "INVALID_START",
                        "the first message must be a JSON start message",
                    )
                )
                await websocket.close(code=1008)
                return
            start, start_error = _parse_start_message(raw_start, server_config)
            if start is None:
                await safe_send(
                    _error_payload(
                        "INVALID_START",
                        start_error or "invalid start message",
                    )
                )
                await websocket.close(code=1008)
                return

            loop = asyncio.get_running_loop()
            model_load_started_at_s = time.perf_counter()
            try:
                session = await loop.run_in_executor(executor, build_session)
                if not callable(getattr(session, "process_frame", None)) or not callable(
                    getattr(session, "reset", None)
                ):
                    raise TypeError("session_factory returned an invalid session")
            except Exception:
                LOGGER.exception("vision session initialization failed")
                await safe_send(
                    _error_payload(
                        "SESSION_INITIALIZATION_FAILED",
                        "vision session could not be initialized",
                    )
                )
                await websocket.close(code=1011)
                return
            factory_elapsed_ms = (time.perf_counter() - model_load_started_at_s) * 1000.0
            model_load_ms = _safe_model_load_ms(session, factory_elapsed_ms)
            LOGGER.info(
                "vision session started session_id=%s source=%sx%s source_fps=%.3f model_load_ms=%.3f",
                start.session_id,
                start.source_width,
                start.source_height,
                start.source_fps,
                model_load_ms,
            )
            worker_task = asyncio.create_task(worker())

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_binary = message.get("bytes")
                raw_text = message.get("text")

                if pending_header is None:
                    if raw_binary is not None:
                        if not await safe_send(
                            _error_payload(
                                "INVALID_MESSAGE_ORDER",
                                "a frame JSON header must precede JPEG binary data",
                            )
                        ):
                            break
                        continue
                    if not isinstance(raw_text, str):
                        if not await safe_send(
                            _error_payload(
                                "INVALID_FRAME_HEADER",
                                "expected a frame JSON header",
                            )
                        ):
                            break
                        continue
                    parsed_header, header_error = _parse_frame_header(raw_text)
                    if parsed_header is None:
                        if not await safe_send(
                            _error_payload(
                                "INVALID_FRAME_HEADER",
                                header_error or "invalid frame header",
                            )
                        ):
                            break
                        continue
                    pending_header = parsed_header
                    continue

                if raw_binary is None:
                    previous_sequence_id = pending_header.sequence_id
                    pending_header = None
                    if not await safe_send(
                        _error_payload(
                            "INVALID_MESSAGE_ORDER",
                            "JPEG binary data must immediately follow its frame header",
                            sequence_id=previous_sequence_id,
                        )
                    ):
                        break
                    # A valid new frame header can begin recovery immediately.
                    if isinstance(raw_text, str):
                        parsed_header, _header_error = _parse_frame_header(raw_text)
                        if parsed_header is not None:
                            pending_header = parsed_header
                    continue

                completed_header = pending_header
                pending_header = None
                received_at_s = time.perf_counter()
                server_received_at_ms = time.time_ns() // 1_000_000
                metrics.received_frames += 1
                if len(raw_binary) > server_config.max_frame_bytes:
                    metrics.rejected_frames += 1
                    if not await safe_send(
                        _error_payload(
                            "FRAME_TOO_LARGE",
                            "JPEG frame exceeds the configured byte limit",
                            sequence_id=completed_header.sequence_id,
                        )
                    ):
                        break
                    continue

                if server_config.max_receive_fps > 0.0:
                    assert metrics.received_at_window is not None
                    while (
                        metrics.received_at_window
                        and received_at_s - metrics.received_at_window[0] >= 1.0
                    ):
                        metrics.received_at_window.popleft()
                    allowed_in_window = max(1, math.ceil(server_config.max_receive_fps))
                    if len(metrics.received_at_window) >= allowed_in_window:
                        metrics.rejected_frames += 1
                        if not await safe_send(
                            _error_payload(
                                "RATE_LIMITED",
                                "frame rate exceeds the configured one-second limit",
                                sequence_id=completed_header.sequence_id,
                            )
                        ):
                            break
                        continue
                    metrics.received_at_window.append(received_at_s)

                pending = _PendingFrame(
                    sequence_id=completed_header.sequence_id,
                    captured_at_ms=completed_header.captured_at_ms,
                    jpeg_bytes=raw_binary,
                    server_received_at_ms=server_received_at_ms,
                    received_at_s=received_at_s,
                )
                if frame_queue.full():
                    replaced = frame_queue.get_nowait()
                    frame_queue.task_done()
                    if replaced is not None:
                        metrics.dropped_frames += 1
                frame_queue.put_nowait(pending)
        finally:
            cancellation: asyncio.CancelledError | None = None
            if worker_task is not None:
                if frame_queue.full():
                    abandoned = frame_queue.get_nowait()
                    frame_queue.task_done()
                    if abandoned is not None:
                        metrics.dropped_frames += 1
                frame_queue.put_nowait(None)
                try:
                    await asyncio.shield(worker_task)
                except asyncio.CancelledError as exc:
                    cancellation = exc
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        LOGGER.exception("vision worker stopped unexpectedly")
                except Exception:
                    LOGGER.exception("vision worker stopped unexpectedly")
            if session is not None:
                reset_future = asyncio.get_running_loop().run_in_executor(
                    executor,
                    session.reset,
                )
                try:
                    await asyncio.shield(reset_future)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                    try:
                        await reset_future
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        LOGGER.exception("vision session reset failed during disconnect")
                except Exception:
                    LOGGER.exception("vision session reset failed during disconnect")
            p50_ms = _percentile(metrics.total_latency_ms or (), 0.50)
            p95_ms = _percentile(metrics.total_latency_ms or (), 0.95)
            LOGGER.info(
                "vision session ended received=%s processed=%s dropped=%s rejected=%s "
                "latency_p50_ms=%s latency_p95_ms=%s elapsed_s=%.3f",
                metrics.received_frames,
                metrics.processed_frames,
                metrics.dropped_frames,
                metrics.rejected_frames,
                round(p50_ms, 3) if p50_ms is not None else None,
                round(p95_ms, 3) if p95_ms is not None else None,
                time.perf_counter() - session_started_at_s,
            )
            gate.release()
            if cancellation is not None:
                raise cancellation

    return application


app = create_app()
