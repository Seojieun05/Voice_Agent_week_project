from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import cv2
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised by installations without server extras
    raise RuntimeError(
        "FastAPI server dependencies are unavailable. Install them with "
        "`pip install -e '.[server]'`."
    ) from exc

from .frame_buffer import BufferedFrame, FrameRingBuffer
from .scene_state import SceneSnapshot, SceneStateStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .chat import ChatClientProtocol
    from .voice import SpeechClientProtocol, TurnDetectorProtocol


LOGGER = logging.getLogger(__name__)


class VisionSessionProtocol(Protocol):
    """Small runtime contract used by the WebSocket server and its fakes."""

    model_load_ms: float

    def process_frame(self, frame: np.ndarray, context: object) -> object: ...

    def reset(self) -> None: ...


SessionFactory = Callable[[], VisionSessionProtocol]
ChatClientFactory = Callable[[], "ChatClientProtocol"]
SpeechClientFactory = Callable[[], "SpeechClientProtocol"]
TurnDetectorFactory = Callable[[], "TurnDetectorProtocol"]


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
    max_question_length: int = 500
    debug_frame_dir: Path | None = None
    vlm_fallback_enabled: bool = True
    vlm_confidence_threshold: float = 0.45
    vlm_cooldown_s: float = 5.0
    vlm_max_image_bytes: int = 1024 * 1024
    vlm_max_image_dim: int = 1024
    frame_buffer_frames: int = 5
    frame_buffer_max_age_s: float = 10.0
    # Endpointing knobs for /ws/audio (lecture 3's "patience dial"):
    # silence_ms of continuous non-speech ends the turn; prefix_ms of audio
    # before onset is kept so first syllables survive; utterances with less
    # than min_speech_ms of actual speech are discarded without an API call.
    silence_ms: int = 700
    prefix_ms: int = 300
    min_speech_ms: int = 250
    vad_aggressiveness: int = 2
    # VAD backend selection: "silero" (neural, probability threshold) with
    # automatic fallback to "webrtc" when silero cannot load. The guard
    # threshold is the raised bar applied while agent TTS may be audible.
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    vad_guard_threshold: float = 0.75
    max_utterance_ms: int = 30_000
    # Barge-in: while a response plays, inbound frames keep feeding the
    # detector (echo-guarded) and user speech onset cancels the response.
    # False restores the legacy behavior of dropping frames during a response.
    barge_in_enabled: bool = True

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
            ("max_question_length", self.max_question_length),
            ("vlm_max_image_bytes", self.vlm_max_image_bytes),
            ("vlm_max_image_dim", self.vlm_max_image_dim),
            ("frame_buffer_frames", self.frame_buffer_frames),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not math.isfinite(self.max_receive_fps) or self.max_receive_fps < 0.0:
            raise ValueError("max_receive_fps must be non-negative")
        if not math.isfinite(self.vlm_confidence_threshold) or not (
            0.0 <= self.vlm_confidence_threshold <= 1.0
        ):
            raise ValueError("vlm_confidence_threshold must be between 0 and 1")
        if not math.isfinite(self.vlm_cooldown_s) or self.vlm_cooldown_s < 0.0:
            raise ValueError("vlm_cooldown_s must be non-negative")
        if not math.isfinite(self.frame_buffer_max_age_s) or self.frame_buffer_max_age_s <= 0.0:
            raise ValueError("frame_buffer_max_age_s must be positive")
        if self.silence_ms < 1:
            raise ValueError("silence_ms must be at least 1")
        if self.prefix_ms < 0:
            raise ValueError("prefix_ms must not be negative")
        if self.min_speech_ms < 0:
            raise ValueError("min_speech_ms must not be negative")
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be between 0 and 3")
        if self.vad_backend.strip().lower() not in {"silero", "webrtc"}:
            raise ValueError("vad_backend must be 'silero' or 'webrtc'")
        for name, value in (
            ("vad_threshold", self.vad_threshold),
            ("vad_guard_threshold", self.vad_guard_threshold),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        if self.vad_guard_threshold < self.vad_threshold:
            raise ValueError("vad_guard_threshold must not be below vad_threshold")
        if self.max_utterance_ms <= self.silence_ms:
            raise ValueError("max_utterance_ms must be greater than silence_ms")
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
            max_question_length=_environment_int("VISION_SERVER_MAX_QUESTION_LENGTH", 500),
            debug_frame_dir=(
                Path(normalized_debug_frame_dir) if normalized_debug_frame_dir else None
            ),
            vlm_fallback_enabled=_environment_bool("VISION_SERVER_VLM_FALLBACK_ENABLED", True),
            vlm_confidence_threshold=_environment_float(
                "VISION_SERVER_VLM_CONFIDENCE_THRESHOLD",
                0.45,
            ),
            vlm_cooldown_s=_environment_float("VISION_SERVER_VLM_COOLDOWN_S", 5.0),
            vlm_max_image_bytes=_environment_int(
                "VISION_SERVER_VLM_MAX_IMAGE_BYTES",
                1024 * 1024,
            ),
            vlm_max_image_dim=_environment_int("VISION_SERVER_VLM_MAX_IMAGE_DIM", 1024),
            frame_buffer_frames=_environment_int("VISION_SERVER_FRAME_BUFFER_FRAMES", 5),
            frame_buffer_max_age_s=_environment_float(
                "VISION_SERVER_FRAME_BUFFER_MAX_AGE_S",
                10.0,
            ),
            silence_ms=_environment_int("VISION_SERVER_SILENCE_MS", 700),
            prefix_ms=_environment_int("VISION_SERVER_PREFIX_MS", 300),
            min_speech_ms=_environment_int("VISION_SERVER_MIN_SPEECH_MS", 250),
            vad_aggressiveness=_environment_int("VISION_SERVER_VAD_AGGRESSIVENESS", 2),
            vad_backend=os.getenv("VISION_SERVER_VAD_BACKEND", "silero"),
            vad_threshold=_environment_float("VISION_SERVER_VAD_THRESHOLD", 0.5),
            vad_guard_threshold=_environment_float(
                "VISION_SERVER_VAD_GUARD_THRESHOLD",
                0.75,
            ),
            max_utterance_ms=_environment_int("VISION_SERVER_MAX_UTTERANCE_MS", 30_000),
            # Deliberately looser than _environment_bool: only an explicit
            # "0"/"false" disables barge-in, so a typo cannot silently turn
            # off the feature the client UX is built around.
            barge_in_enabled=(
                os.getenv("VISION_SERVER_BARGE_IN", "1").strip().lower() not in {"0", "false"}
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
    frame_width: int = 0
    frame_height: int = 0


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


async def _wait_shielded(
    future: "asyncio.Future[None]",
    failure_message: str,
) -> asyncio.CancelledError | None:
    """Wait for cleanup work to finish even while this task is being cancelled.

    A bare ``await future`` is unsafe during cancellation: cancelling a task
    also cancels the future it is currently awaiting, and an executor future
    that has not started yet is then discarded without ever running.
    Cancellation can also be delivered repeatedly (anyio cancel scopes
    re-cancel at every checkpoint), so every retry must be shielded again,
    not just the first attempt. Returns the first observed cancellation so
    the caller can re-raise it once cleanup is complete.
    """
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = exc
            if future.cancelled():
                return cancellation
            continue
        except Exception:
            LOGGER.exception(failure_message)
        return cancellation


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
        frame_width=width,
        frame_height=height,
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


_MAX_VISIBLE_OBJECTS = 12
# Weak detections without analyzer confirmation are noise for the chat AI.
_MIN_VISIBLE_CONFIDENCE = 0.30
# Below this, an unconfirmed detection is flagged so answers can hedge once.
_UNCERTAIN_VISIBLE_CONFIDENCE = 0.50


def _visible_object_importance(object_type: str) -> int:
    """Rank object types by how much they matter to a walking pedestrian."""
    normalized = object_type.strip().lower().replace(" ", "_")
    if "signal" in normalized or "traffic_light" in normalized:
        return 5
    if normalized in {"bus", "car", "truck", "van", "motorcycle", "train", "scooter"}:
        return 4
    if normalized in {"person", "bicycle", "wheelchair", "stroller", "dog"}:
        return 3
    for marker in ("kiosk", "sign", "screen", "panel", "text", "door", "stairs"):
        if marker in normalized:
            return 2
    return 1


def _visible_object_text(attributes: Mapping[str, object]) -> str | None:
    """Extract confirmed OCR text (sign text, bus route, kiosk lines) if any."""
    for key in ("text", "route_number"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    visible_text = attributes.get("visible_text")
    if isinstance(visible_text, Sequence) and not isinstance(
        visible_text,
        (str, bytes, bytearray),
    ):
        lines = [str(line).strip() for line in visible_text if str(line).strip()]
        if lines:
            return " / ".join(lines[:5])
    return None


def _position_label(left: float, right: float, frame_width: int) -> str | None:
    """Describe where an object sits horizontally, by overlap not center.

    A large object (e.g. furniture filling the right half) can have its
    center near the middle while actually occupying one side; the third
    with the largest overlap is what a walking user experiences.
    """
    if frame_width <= 0:
        return None
    left = max(0.0, min(left, float(frame_width)))
    right = max(0.0, min(right, float(frame_width)))
    if right <= left:
        return None
    if (right - left) >= frame_width * 0.75:
        return "전방 전체"
    third = frame_width / 3.0
    overlaps = {
        "왼쪽": max(0.0, min(right, third) - left),
        "중앙": max(0.0, min(right, 2.0 * third) - max(left, third)),
        "오른쪽": max(0.0, right - max(left, 2.0 * third)),
    }
    return max(overlaps, key=overlaps.__getitem__)


def _distance_label(
    left: float,
    right: float,
    top: float,
    bottom: float,
    frame_width: int,
    frame_height: int,
) -> str | None:
    """Estimate proximity from how much of the frame the object fills.

    A far-away pedestrian occupies a small box and must not read as an
    obstacle; a nearby one fills much of the frame height.
    """
    if frame_width <= 0 or frame_height <= 0:
        return None
    height_ratio = max(0.0, bottom - top) / frame_height
    width_ratio = max(0.0, right - left) / frame_width
    if height_ratio <= 0.0 or width_ratio <= 0.0:
        return None
    area_ratio = height_ratio * width_ratio
    bottom_ratio = min(bottom, float(frame_height)) / frame_height
    # "가까움" must mean imminent (roughly within arm's-to-two-steps reach):
    # a pedestrian a few meters away already fills half the frame height, so
    # close requires a dominant box or one that reaches the frame bottom.
    if height_ratio >= 0.8 or area_ratio >= 0.4 or (height_ratio >= 0.55 and bottom_ratio >= 0.85):
        return "가까움"
    if height_ratio >= 0.35 or area_ratio >= 0.12:
        return "중간"
    return "멀리"


def _serialized_visible_objects(
    analysis: object,
    frame_width: int,
    frame_height: int,
) -> tuple[list[dict[str, object]], float | None, list[dict[str, object]]]:
    """Summarize what is currently visible for the chat scene state.

    Unlike ``analysis_events`` (state changes only), this reflects every
    detection in the latest frame, enriched with analyzer state and confirmed
    OCR text but stripped of internal debug attributes. Also returns the best
    detection confidence of the kept objects (for the VLM fallback) and the
    full raw detection list (bbox, track id, raw confidence) for the chat
    tools — neither of which is sent to the chat model directly.
    """
    detections = getattr(analysis, "detections", ())
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes, bytearray)):
        return [], None, []
    results_by_index = getattr(analysis, "analysis_results_by_index", {})
    if not isinstance(results_by_index, Mapping):
        results_by_index = {}

    objects: list[tuple[int, bool, float, dict[str, object]]] = []
    raw_detections: list[dict[str, object]] = []
    for index, detection in enumerate(detections):
        object_type = str(getattr(detection, "class_name", "")).strip()
        if not object_type:
            continue
        entry: dict[str, object] = {"object_type": object_type}
        confidence = _finite_number(getattr(detection, "confidence", None))

        position: str | None = None
        distance: str | None = None
        bbox: list[float] | None = None
        size_percent: float | None = None
        left = right = top = bottom = None
        xyxy = getattr(detection, "xyxy", None)
        if (
            frame_width > 0
            and isinstance(xyxy, Sequence)
            and not isinstance(xyxy, (str, bytes, bytearray))
            and len(xyxy) == 4
        ):
            left = _finite_number(xyxy[0])
            right = _finite_number(xyxy[2])
            top = _finite_number(xyxy[1])
            bottom = _finite_number(xyxy[3])
            if left is not None and right is not None:
                bbox = [
                    round(left, 1),
                    round(top, 1) if top is not None else 0.0,
                    round(right, 1),
                    round(bottom, 1) if bottom is not None else 0.0,
                ]
                position = _position_label(left, right, frame_width)
                if top is not None and bottom is not None:
                    distance = _distance_label(
                        left,
                        right,
                        top,
                        bottom,
                        frame_width,
                        frame_height,
                    )
                    if frame_height > 0:
                        size_percent = round(
                            max(0.0, right - left)
                            * max(0.0, bottom - top)
                            / (frame_width * frame_height)
                            * 100.0,
                            1,
                        )
                    # A frame-wide but not-close box (a train or bridge in
                    # the background) does not block the path; "전방 전체"
                    # is reserved for objects the user could walk into.
                    if position == "전방 전체" and distance != "가까움":
                        position = "중앙"

        raw_entry: dict[str, object] = {"class_name": object_type}
        if confidence is not None:
            raw_entry["confidence"] = round(confidence, 3)
        if bbox is not None:
            raw_entry["bbox_xyxy"] = bbox
        track_id = getattr(detection, "track_id", None)
        if isinstance(track_id, int) and not isinstance(track_id, bool):
            raw_entry["track_id"] = track_id
        if position is not None:
            raw_entry["direction"] = position
        if distance is not None:
            raw_entry["distance"] = distance
        if size_percent is not None:
            raw_entry["size_percent"] = size_percent
        raw_detections.append(raw_entry)

        # A "person" whose box starts low in the frame and runs off the
        # bottom edge is the user's own feet/legs entering the shot, not
        # someone standing in front (a real close person shows a head in
        # the upper half). Reporting it caused "앞에 사람이 있어요".
        if (
            object_type.lower() == "person"
            and top is not None
            and bottom is not None
            and frame_height > 0
            and top >= frame_height * 0.55
            and bottom >= frame_height * 0.95
        ):
            continue
        if position is not None:
            entry["position"] = position
        if distance is not None:
            entry["distance"] = distance

        result = results_by_index.get(index)
        if result is not None:
            result_type = str(getattr(result, "object_type", "")).strip()
            if result_type:
                entry["object_type"] = result_type
            state = getattr(result, "state", None)
            if state is not None:
                state_text = str(getattr(state, "value", state)).strip()
                # Placeholder analyzer states are internal bookkeeping, not
                # information; forwarding them made every pedestrian carry
                # "state": "UNKNOWN" and an uncertainty flag.
                if state_text and state_text.upper() not in {"UNKNOWN", "DESCRIBED"}:
                    entry["state"] = state_text
            attributes = getattr(result, "attributes", None)
            if isinstance(attributes, Mapping):
                text = _visible_object_text(attributes)
                if text is not None:
                    entry["text"] = text
                description = attributes.get("description")
                if isinstance(description, str) and description.strip():
                    entry["description"] = description.strip()

        confirmed = "state" in entry or "text" in entry
        if confirmed:
            # Analyzer-level uncertainty only matters alongside an actual
            # analyzer finding (a signal color, confirmed text).
            if result is not None and bool(getattr(result, "is_uncertain", False)):
                entry["is_uncertain"] = True
        elif confidence is not None:
            # Raw confidence numbers push the chat model into hedging every
            # sentence; keep only a boolean flag for genuinely weak detections
            # and drop unconfirmed noise entirely.
            if confidence < _MIN_VISIBLE_CONFIDENCE:
                continue
            if confidence < _UNCERTAIN_VISIBLE_CONFIDENCE:
                entry["is_uncertain"] = True
        importance = _visible_object_importance(str(entry["object_type"]))
        objects.append((importance, confirmed, confidence or 0.0, entry))

    # Importance-first ordering: the first entry is the one the answer
    # should describe in detail, and truncation drops only low-signal items.
    objects.sort(key=lambda item: item[:3], reverse=True)
    kept = objects[:_MAX_VISIBLE_OBJECTS]
    scene_confidence = max((confidence for _, _, confidence, _ in kept), default=None)
    return [entry for _, _, _, entry in kept], scene_confidence, raw_detections


# Korean aliases so find_object("버스") works without the model translating.
_OBJECT_NAME_ALIASES = {
    "사람": "person",
    "버스": "bus",
    "자동차": "car",
    "차": "car",
    "트럭": "truck",
    "자전거": "bicycle",
    "오토바이": "motorcycle",
    "신호등": "traffic_light",
    "보행자신호": "pedestrian_signal",
    "표지판": "traffic_sign",
    "볼라드": "bollard",
    "나무": "tree_trunk",
    "가로수": "tree_trunk",
    "기둥": "pole",
    "전봇대": "pole",
    "의자": "chair",
    "벤치": "bench",
    "키오스크": "kiosk",
    "휠체어": "wheelchair",
    "유모차": "stroller",
    "소화전": "fire_hydrant",
    "입간판": "movable_signage",
    "간판": "movable_signage",
    "킥보드": "scooter",
    "스쿠터": "scooter",
}


def _normalize_object_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    return _OBJECT_NAME_ALIASES.get(name.strip(), normalized)


def _seconds_since(updated_at_ms: int | None) -> float | None:
    if updated_at_ms is None:
        return None
    now_ms = time.time_ns() // 1_000_000
    return max(0.0, round((now_ms - updated_at_ms) / 1000.0, 1))


def _tool_get_current_scene(snapshot: SceneSnapshot) -> dict[str, object]:
    """Chat tool: full latest-frame YOLO detections with metadata."""
    return {
        "detected_objects": [dict(item) for item in snapshot.raw_detections],
        "object_count": len(snapshot.raw_detections),
        "detected_at_ms": snapshot.updated_at_ms,
        "seconds_since_detection": _seconds_since(snapshot.updated_at_ms),
    }


def _tool_find_object(snapshot: SceneSnapshot, name: str) -> dict[str, object]:
    """Chat tool: whether a specific object class is currently visible."""
    query = _normalize_object_name(name)
    if not query:
        return {"error": "name_required"}
    matches = []
    for item in snapshot.raw_detections:
        class_name = str(item.get("class_name", "")).strip().lower().replace(" ", "_")
        if query == class_name or query in class_name or class_name in query:
            match: dict[str, object] = {"class_name": item.get("class_name")}
            for key in ("direction", "distance", "size_percent", "confidence", "track_id"):
                if key in item:
                    match[key] = item[key]
            matches.append(match)
    return {
        "query": name,
        "found": bool(matches),
        "matches": matches,
        "seconds_since_detection": _seconds_since(snapshot.updated_at_ms),
    }


def _tool_check_traffic_light(snapshot: SceneSnapshot) -> dict[str, object]:
    """Chat tool: current signal state, judgement confidence, last update."""

    def _is_signal(type_name: object) -> bool:
        normalized = str(type_name).strip().lower().replace(" ", "_")
        return "signal" in normalized or "traffic_light" in normalized

    visible = [item for item in snapshot.visible_objects if _is_signal(item.get("object_type"))]
    raw = [item for item in snapshot.raw_detections if _is_signal(item.get("class_name"))]
    state = None
    is_uncertain = False
    for item in visible:
        if "state" in item:
            state = item["state"]
            is_uncertain = bool(item.get("is_uncertain", False))
            break
    confidences = [
        float(item["confidence"])
        for item in raw
        if isinstance(item.get("confidence"), (int, float))
    ]
    signal_events = [
        dict(event) for event in snapshot.recent_events if _is_signal(event.get("object_type", ""))
    ]
    return {
        "traffic_light_visible": bool(visible or raw),
        "state": state if state is not None else "UNKNOWN",
        "is_uncertain": is_uncertain,
        "detection_confidence": max(confidences) if confidences else None,
        "last_update_ms": snapshot.updated_at_ms,
        "seconds_since_update": _seconds_since(snapshot.updated_at_ms),
        "recent_signal_events": signal_events,
    }


def _tool_get_recent_changes(snapshot: SceneSnapshot) -> dict[str, object]:
    """Chat tool: what changed recently (events + spoken narrations)."""
    return {
        "recent_events": [dict(event) for event in snapshot.recent_events],
        "recent_narrations": list(snapshot.latest_narrations),
        "seconds_since_last_frame": _seconds_since(snapshot.updated_at_ms),
        "note": "recent_events의 seconds_ago는 몇 초 전 발생했는지를 뜻합니다",
    }


def _tool_read_detected_text(snapshot: SceneSnapshot) -> dict[str, object]:
    """Chat tool: OCR-confirmed text currently visible, without a VLM call."""
    texts = []
    for item in snapshot.visible_objects:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            entry: dict[str, object] = {
                "object_type": item.get("object_type"),
                "text": text.strip(),
            }
            if "position" in item:
                entry["direction"] = item["position"]
            texts.append(entry)
    return {
        "texts": texts,
        "found": bool(texts),
        "seconds_since_detection": _seconds_since(snapshot.updated_at_ms),
        "note": (
            "OCR로 확정된 글자만 포함됩니다. 비어 있으면 analyze_frame_with_vlm으로 "
            "이미지에서 직접 읽어볼 수 있습니다"
        ),
    }


def _tool_check_camera_status(
    snapshot: SceneSnapshot,
    buffered_frames: Sequence[BufferedFrame],
    stream_active: bool,
) -> dict[str, object]:
    """Chat tool: is the camera stream healthy and how fresh is the data."""
    newest_frame_age_s = None
    if buffered_frames:
        now_ms = time.time_ns() // 1_000_000
        newest_ms = max(frame.received_at_ms for frame in buffered_frames)
        newest_frame_age_s = max(0.0, round((now_ms - newest_ms) / 1000.0, 1))
    return {
        "stream_connected": stream_active,
        "has_any_analysis": snapshot.has_analysis,
        "seconds_since_last_analysis": _seconds_since(snapshot.updated_at_ms),
        "buffered_frame_count": len(buffered_frames),
        "newest_frame_age_s": newest_frame_age_s,
    }


_VLM_DETAIL_KEYWORDS = ("자세히", "상세", "구체적", "묘사")
_VLM_VISION_KEYWORDS = (
    "색",
    "글자",
    "글씨",
    "뭐라",
    "적혀",
    "써 있",
    "써있",
    "쓰여",
    "읽어",
    "그림",
    "모양",
    "생겼",
    "모습",
    "입은",
    "입고",
    "표정",
)


def _vlm_trigger_reason(
    snapshot: SceneSnapshot,
    question: str,
    config: ServerConfig,
) -> str | None:
    """Decide whether this question warrants the Grok Vision fallback."""
    if not config.vlm_fallback_enabled:
        return None
    if not snapshot.visible_objects:
        return "no_detections"
    if (
        snapshot.scene_confidence is not None
        and snapshot.scene_confidence < config.vlm_confidence_threshold
    ):
        return "low_confidence"
    if any(keyword in question for keyword in _VLM_VISION_KEYWORDS):
        return "question_needs_vision"
    if any(keyword in question for keyword in _VLM_DETAIL_KEYWORDS):
        return "detail_requested"
    from .chat import classify_question

    question_type = classify_question(question)
    # Go/no-go and crossing judgments need the image: path/sidewalk presence
    # and signal countdown digits are not in the YOLO object list.
    if question_type in {"can_i_go", "crossing"}:
        return "path_check"
    # A describe answer built only from far-away unlabeled objects says
    # nothing about the actual surroundings (river, park, buildings) —
    # the image is the only informative source then.
    if question_type == "describe" and not any(
        item.get("distance") != "멀리" or "state" in item or "text" in item
        for item in snapshot.visible_objects
    ):
        return "sparse_scene"
    return None


def _frame_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Sharpness comparison must never trade recency for clarity: only frames
# this close to the newest one may compete, so answers describe "now".
_VLM_CANDIDATE_WINDOW_MS = 1500


def _select_vlm_image(frames: Sequence[BufferedFrame], config: ServerConfig) -> bytes | None:
    """Pick the sharpest of the freshest frames and bound its size.

    CPU-bound (JPEG decode + Laplacian); run off the event loop. Returns
    re-encoded JPEG bytes within the configured limits, or None when no
    buffered frame decodes.
    """
    if not frames:
        return None
    newest_ms = max(frame.received_at_ms for frame in frames)
    candidates = [
        frame for frame in frames if newest_ms - frame.received_at_ms <= _VLM_CANDIDATE_WINDOW_MS
    ]
    best_frame: np.ndarray | None = None
    best_sharpness = -1.0
    # Newest-first so ties prefer the most recent frame.
    for buffered in sorted(candidates, key=lambda frame: frame.received_at_ms, reverse=True)[:3]:
        encoded = np.frombuffer(buffered.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            continue
        sharpness = _frame_sharpness(frame)
        if sharpness > best_sharpness:
            best_frame = frame
            best_sharpness = sharpness
    if best_frame is None:
        return None

    height, width = best_frame.shape[:2]
    longest = max(height, width)
    if longest > config.vlm_max_image_dim:
        scale = config.vlm_max_image_dim / float(longest)
        best_frame = cv2.resize(
            best_frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    for quality in (85, 70, 55):
        ok, encoded_jpeg = cv2.imencode(
            ".jpg",
            best_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None
        if len(encoded_jpeg) <= config.vlm_max_image_bytes:
            return encoded_jpeg.tobytes()
    return None


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


class _ChatRequest(BaseModel):
    session_id: str
    user_question: str


def _default_chat_client_factory() -> "ChatClientProtocol":
    # Chat dependencies stay optional for vision-only installations, so the
    # Grok client is imported only when the chat API is actually used.
    from .chat import GrokChatClient

    return GrokChatClient.from_environment()


def _default_speech_client_factory() -> "SpeechClientProtocol":
    # Speech dependencies stay optional for vision-only installations, so the
    # client is imported only when the audio WebSocket is actually used.
    from .voice import GrokSpeechClient

    return GrokSpeechClient.from_environment()


# Second echo-defense layer for barge-in (after the raised VAD bar): a
# barged-in "turn" made mostly of the agent's own words is playback bleeding
# into the mic, not the user. 0.6 rather than higher because echo transcripts
# routinely lose a word or two to the guard threshold and STT noise.
_ECHO_OVERLAP_THRESHOLD = 0.6

# audio_end 후 이 시간 안에 playback_done이 오지 않으면 턴을 강제 종결한다.
# 답변 재생은 길어야 수십 초다. 짧게 잡으면 정상 재생 중에 마이크를 열어
# 자기 에코를 받으므로 넉넉하게 둔다.
PLAYBACK_DONE_TIMEOUT_S = 60.0


def _echo_tokens(text: str) -> set[str]:
    # \w+ tokenization instead of split(): the reply carries punctuation glued
    # to words ("있습니다.") while STT output does not ("있습니다") — split()
    # would systematically undercount overlap on exactly the short echoes
    # where this defense matters most.
    return set(re.findall(r"\w+", text.casefold()))


# Below this many transcript tokens the overlap ratio is too coarse to judge:
# a one-word follow-up that reuses a reply word ("오른쪽?") scores 1.0, and a
# genuine question would be silently swallowed right after its own barge-in
# killed the playback. Letting a rare 1-2 word echo through costs one odd
# chat call; eating a real question costs the user's trust.
_ECHO_MIN_TOKENS = 3


def _transcript_echo_overlap(transcript: str, reply: str) -> float:
    """Fraction of the transcript's token set that also appears in the reply.

    Empty inputs and transcripts shorter than [_ECHO_MIN_TOKENS] tokens can
    never be judged as echo, hence 0.0.
    """
    transcript_words = _echo_tokens(transcript)
    reply_words = _echo_tokens(reply)
    if len(transcript_words) < _ECHO_MIN_TOKENS or not reply_words:
        return 0.0
    return len(transcript_words & reply_words) / len(transcript_words)


def _parse_audio_start_message(
    raw_text: str,
    config: ServerConfig,
) -> tuple[str | None, str | None]:
    """Parse the /ws/audio start message; returns (session_id, error)."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, "start message is not valid JSON"
    if not isinstance(payload, dict) or payload.get("type") != "start":
        return None, "the first message must have type 'start'"
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None, "session_id must be a non-empty string"
    session_id = session_id.strip()
    if len(session_id) > config.max_session_id_length:
        return None, "session_id exceeds the configured maximum length"
    return session_id, None


def create_app(
    config: ServerConfig | None = None,
    session_factory: SessionFactory | None = None,
    chat_client_factory: ChatClientFactory | None = None,
    scene_store: SceneStateStore | None = None,
    speech_client_factory: SpeechClientFactory | None = None,
    turn_detector_factory: TurnDetectorFactory | None = None,
) -> FastAPI:
    """Create an injectable single-session FastAPI application."""
    server_config = config or ServerConfig.from_environment()
    build_session = session_factory or _default_session_factory(server_config)
    build_chat_client = chat_client_factory or _default_chat_client_factory
    build_speech_client = speech_client_factory or _default_speech_client_factory
    store = scene_store or SceneStateStore()
    frame_buffer = FrameRingBuffer(max_frames=server_config.frame_buffer_frames)
    vlm_last_called: dict[str, float] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-worker")
    gate = _SingleSessionGate()
    audio_gate = _SingleSessionGate()
    chat_client: "ChatClientProtocol | None" = None
    chat_client_guard = asyncio.Lock()
    speech_client: "SpeechClientProtocol | None" = None
    speech_client_guard = asyncio.Lock()

    def _default_turn_detector() -> "TurnDetectorProtocol":
        # VAD dependencies (silero-vad/webrtcvad) stay optional for
        # installations that never use audio, hence the local imports.
        from .vad_backends import create_vad
        from .voice import TurnDetector

        vad = create_vad(
            server_config.vad_backend,
            aggressiveness=server_config.vad_aggressiveness,
            threshold=server_config.vad_threshold,
            guard_threshold=server_config.vad_guard_threshold,
        )
        return TurnDetector(
            silence_ms=server_config.silence_ms,
            prefix_ms=server_config.prefix_ms,
            min_speech_ms=server_config.min_speech_ms,
            max_utterance_ms=server_config.max_utterance_ms,
            vad=vad,
        )

    build_turn_detector = turn_detector_factory or _default_turn_detector

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        yield
        executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(chat_client, "close", None)
        if callable(close):
            close()
        speech_close = getattr(speech_client, "close", None)
        if callable(speech_close):
            speech_close()

    application = FastAPI(title="Voice Agent Vision Server", lifespan=lifespan)
    application.state.server_config = server_config
    application.state.session_gate = gate
    application.state.scene_store = store

    async def _chat_client() -> "ChatClientProtocol":
        nonlocal chat_client
        async with chat_client_guard:
            if chat_client is None:
                chat_client = build_chat_client()
            return chat_client

    async def _speech_client() -> "SpeechClientProtocol":
        nonlocal speech_client
        async with speech_client_guard:
            if speech_client is None:
                speech_client = build_speech_client()
            return speech_client

    @application.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "active_session": gate.active,
        }

    @application.post("/api/session")
    async def create_session() -> dict[str, object]:
        session_id = uuid4().hex
        store.register(session_id)
        return {
            "session_id": session_id,
            "created_at_ms": time.time_ns() // 1_000_000,
        }

    async def answer_question(session_id: str, question: str) -> tuple[int, dict[str, object]]:
        """Answer one user question against the session's latest scene state.

        Shared by POST /api/chat and /ws/audio voice turns. Returns an
        HTTP-style ``(status_code, payload)`` pair so both callers map
        success and errors identically.
        """
        from .chat import ChatServiceError

        snapshot = store.snapshot(session_id)
        if snapshot is None:
            return 404, _error_payload(
                "SESSION_NOT_FOUND",
                "unknown session_id; create one via POST /api/session "
                "or start a /ws/vision stream first",
            )

        scene_state = snapshot.to_dict()
        if snapshot.updated_at_ms is not None:
            now_ms = time.time_ns() // 1_000_000
            scene_state["seconds_since_last_frame"] = max(
                0,
                round((now_ms - snapshot.updated_at_ms) / 1000.0),
            )

        trigger = _vlm_trigger_reason(snapshot, question, server_config)
        vlm_meta: dict[str, object] = {"used": False, "reason": None, "latency_ms": None}
        tool_calls_meta: list[dict[str, object]] = []

        def run_vlm_tool(tool_client: "ChatClientProtocol", vlm_question: str) -> dict[str, object]:
            """analyze_frame_with_vlm tool body; runs inside the tool thread."""
            create_vision = getattr(tool_client, "create_vision_answer", None)
            if not callable(create_vision):
                return {"error": "vlm_unavailable"}
            if not server_config.vlm_fallback_enabled:
                return {"error": "vlm_disabled"}
            now_s = time.monotonic()
            last_called_s = vlm_last_called.get(session_id)
            if last_called_s is not None and now_s - last_called_s < server_config.vlm_cooldown_s:
                return {
                    "error": "cooldown_active",
                    "retry_after_s": round(
                        server_config.vlm_cooldown_s - (now_s - last_called_s),
                        1,
                    ),
                }
            frames = frame_buffer.recent(
                session_id,
                max_age_ms=int(server_config.frame_buffer_max_age_s * 1000.0),
            )
            if not frames:
                return {"error": "no_recent_frame"}
            image = _select_vlm_image(frames, server_config)
            if image is None:
                return {"error": "no_usable_frame"}
            # Stamp before calling so failures also respect the cooldown.
            vlm_last_called[session_id] = now_s
            started_s = time.perf_counter()
            try:
                answer = create_vision(scene_state, vlm_question, image)
            except ChatServiceError as exc:
                LOGGER.warning("vlm tool failed code=%s", exc.code)
                return {"error": f"vlm_failed:{exc.code}"}
            latency_ms = round((time.perf_counter() - started_s) * 1000.0, 1)
            vlm_meta.update({"used": True, "reason": "tool_call", "latency_ms": latency_ms})
            return {"vlm_answer": answer}

        def make_tool_executor(tool_client: "ChatClientProtocol"):
            def execute_tool(name: str, arguments: Mapping[str, object]) -> object:
                if name == "get_current_scene":
                    return _tool_get_current_scene(snapshot)
                if name == "find_object":
                    return _tool_find_object(snapshot, str(arguments.get("name", "")))
                if name == "check_traffic_light":
                    return _tool_check_traffic_light(snapshot)
                if name == "get_recent_changes":
                    return _tool_get_recent_changes(snapshot)
                if name == "read_detected_text":
                    return _tool_read_detected_text(snapshot)
                if name == "check_camera_status":
                    return _tool_check_camera_status(
                        snapshot,
                        frame_buffer.recent(session_id),
                        gate.active,
                    )
                if name == "analyze_frame_with_vlm":
                    vlm_question = str(arguments.get("question", "")).strip() or question
                    return run_vlm_tool(tool_client, vlm_question)
                return {"error": f"unknown_tool:{name}"}

            return execute_tool

        async def try_vlm_answer(vlm_client: "ChatClientProtocol") -> str | None:
            """Attempt the vision fallback; on any miss record why and return None."""
            create_vision = getattr(vlm_client, "create_vision_answer", None)
            if not callable(create_vision):
                vlm_meta["reason"] = "client_unsupported"
                return None
            now_s = time.monotonic()
            last_called_s = vlm_last_called.get(session_id)
            if last_called_s is not None and now_s - last_called_s < server_config.vlm_cooldown_s:
                vlm_meta["reason"] = "cooldown_active"
                return None
            frames = frame_buffer.recent(
                session_id,
                max_age_ms=int(server_config.frame_buffer_max_age_s * 1000.0),
            )
            if not frames:
                vlm_meta["reason"] = "no_recent_frame"
                return None
            image = await asyncio.to_thread(_select_vlm_image, frames, server_config)
            if image is None:
                vlm_meta["reason"] = "no_usable_frame"
                return None
            # Stamp before calling so failures also respect the cooldown.
            vlm_last_called[session_id] = now_s
            if len(vlm_last_called) > 512:
                for stale_key, _ in sorted(vlm_last_called.items(), key=lambda kv: kv[1])[:256]:
                    vlm_last_called.pop(stale_key, None)
            started_s = time.perf_counter()
            try:
                answer = await asyncio.to_thread(create_vision, scene_state, question, image)
            except ChatServiceError as exc:
                vlm_meta["reason"] = f"vlm_failed:{exc.code}"
                LOGGER.warning("vlm fallback failed trigger=%s code=%s", trigger, exc.code)
                return None
            latency_ms = round((time.perf_counter() - started_s) * 1000.0, 1)
            vlm_meta.update({"used": True, "reason": trigger, "latency_ms": latency_ms})
            LOGGER.info("vlm fallback used trigger=%s latency_ms=%s", trigger, latency_ms)
            return answer

        try:
            client = await _chat_client()
            answer_text: str | None = None
            create_tool = getattr(client, "create_tool_answer", None)
            if callable(create_tool):
                # Tool-calling path: Grok decides which scene tools to call.
                answer_text, tool_calls_meta = await asyncio.to_thread(
                    create_tool,
                    scene_state,
                    question,
                    make_tool_executor(client),
                )
                if tool_calls_meta:
                    LOGGER.info(
                        "chat tools used=%s",
                        [call["name"] for call in tool_calls_meta],
                    )
            else:
                # Legacy path for injected clients without tool support.
                if trigger is not None:
                    answer_text = await try_vlm_answer(client)
                if answer_text is None:
                    answer_text = await asyncio.to_thread(
                        client.create_answer,
                        scene_state,
                        question,
                    )
        except ChatServiceError as exc:
            # Configuration problems are the operator's to fix (503); the rest
            # are upstream failures (502). The connection must survive both.
            status_code = 503 if exc.code == "MISSING_API_KEY" else 502
            LOGGER.warning("chat answer failed code=%s", exc.code)
            return status_code, _error_payload(exc.code, exc.message)
        except Exception:
            LOGGER.exception("chat answer failed unexpectedly")
            return 500, _error_payload("CHAT_FAILED", "chat answer could not be generated")

        return 200, {
            "type": "chat_answer",
            "session_id": session_id,
            "answer_text": answer_text,
            "has_scene_analysis": snapshot.has_analysis,
            "scene_state_updated_at_ms": snapshot.updated_at_ms,
            "vlm": vlm_meta,
            "tool_calls": tool_calls_meta,
        }

    @application.post("/api/chat")
    async def chat(request: _ChatRequest) -> JSONResponse:
        session_id = request.session_id.strip()
        question = request.user_question.strip()
        if not session_id or len(session_id) > server_config.max_session_id_length:
            return JSONResponse(
                status_code=400,
                content=_error_payload("INVALID_SESSION_ID", "session_id is empty or too long"),
            )
        if not question:
            return JSONResponse(
                status_code=400,
                content=_error_payload("INVALID_QUESTION", "user_question must not be empty"),
            )
        if len(question) > server_config.max_question_length:
            return JSONResponse(
                status_code=400,
                content=_error_payload(
                    "QUESTION_TOO_LONG",
                    "user_question exceeds the configured maximum length",
                ),
            )
        status_code, payload = await answer_question(session_id, question)
        return JSONResponse(status_code=status_code, content=payload)

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

        async def worker(vision_session_id: str) -> None:
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
                    serialized_events = _serialized_events(getattr(analysis, "analysis_events", ()))
                    serialized_narrations = _serialized_narrations(
                        getattr(analysis, "narrations", ())
                    )
                    completed_at_ms = time.time_ns() // 1_000_000
                    visible_objects, scene_confidence, raw_detections = _serialized_visible_objects(
                        analysis,
                        outcome.frame_width,
                        outcome.frame_height,
                    )
                    store.update(
                        vision_session_id,
                        analysis_events=serialized_events,
                        narrations=serialized_narrations,
                        visible_objects=visible_objects,
                        scene_confidence=scene_confidence,
                        raw_detections=raw_detections,
                        updated_at_ms=completed_at_ms,
                    )
                    response: dict[str, object] = {
                        "type": "analysis",
                        "sequence_id": pending.sequence_id,
                        "captured_at_ms": pending.captured_at_ms,
                        "server_received_at_ms": pending.server_received_at_ms,
                        "completed_at_ms": completed_at_ms,
                        "dropped_frames": metrics.dropped_frames,
                        "received_frames": metrics.received_frames,
                        "processed_frames": metrics.processed_frames,
                        "processing_fps": round(metrics.processing_fps(), 3),
                        "model_load_ms": round(model_load_ms, 3),
                        "analysis_events": serialized_events,
                        "narrations": serialized_narrations,
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
            store.register(start.session_id)
            worker_task = asyncio.create_task(worker(start.session_id))

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

                # Keep the accepted JPEG for potential VLM fallback use;
                # memory only, bounded by the per-session ring buffer.
                frame_buffer.append(
                    start.session_id,
                    raw_binary,
                    sequence_id=completed_header.sequence_id,
                    received_at_ms=server_received_at_ms,
                )
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
                cancellation = await _wait_shielded(
                    worker_task,
                    "vision worker stopped unexpectedly",
                )
            if session is not None:
                reset_future = asyncio.get_running_loop().run_in_executor(
                    executor,
                    session.reset,
                )
                reset_cancellation = await _wait_shielded(
                    reset_future,
                    "vision session reset failed during disconnect",
                )
                if cancellation is None:
                    cancellation = reset_cancellation
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

    @application.websocket("/ws/audio")
    async def audio_websocket(websocket: WebSocket) -> None:
        """Hands-free voice turns: PCM frames in, transcript + TTS chunks out.

        Wire protocol (all control messages are JSON text; audio is binary):

            client -> server  {"type": "start", "session_id": str}   (first)
            client -> server  <binary>  640-byte int16 LE PCM frame (20 ms @ 16 kHz)
            client -> server  {"type": "playback_done"}   (speaker truly done)
            server -> client  {"type": "ready", "session_id": str}
            server -> client  {"type": "vad", "speaking": bool}      (UI only)
            server -> client  {"type": "turn", "duration_ms": int}
            server -> client  {"type": "transcript", "text": str}
            server -> client  {"type": "audio_start"}
            server -> client  <binary>  one encoded TTS chunk (mp3), repeated
            server -> client  {"type": "audio_end", "reply": str,
                               "tool_calls": [...], "timings": {...}}
            server -> client  {"type": "interrupted"}   (barge-in, see below)
            server -> client  {"type": "listening"} / {"type": "error", ...}

        Barge-in (default): while a response is in flight, inbound frames
        keep feeding the same turn detector with its echo guard raised (the
        VAD applies its stricter bar so playback bleed rarely reads as
        speech). The moment user speech starts anyway, the server sends
        ``interrupted`` once, cancels the respond task (no ``audio_end``),
        and the utterance flows into the normal turn pipeline. A second echo
        defense drops barged-in turns whose transcript is mostly the words
        of the reply that was playing. ``playback_done`` is ignored while
        the detector reports speech so the mark of an interrupted playback
        cannot reset a turn in progress. With ``barge_in_enabled=False``
        inbound audio is simply dropped during a response (legacy behavior).
        """
        from .voice import FRAME_BYTES, SpeechServiceError

        await websocket.accept()
        if not await audio_gate.claim():
            await websocket.send_json(
                _error_payload(
                    "SESSION_BUSY",
                    "another audio session is already active",
                )
            )
            await websocket.close(code=1013)
            return

        detector: "TurnDetectorProtocol | None" = None
        respond_task: asyncio.Task[None] | None = None
        send_lock = asyncio.Lock()
        responding = False
        early_playback_done = False
        # Barge-in bookkeeping: whether the turn currently accumulating in
        # the detector started while a response was playing, and the reply
        # text that was on the speaker then (for the echo overlap check).
        barged_in = False
        last_reply: str | None = None
        # audio_end 전송 시각. playback_done이 오지 않는 클라이언트(중간에 재생을
        # 끊고 mark를 잃는 등)로부터 responding 상태가 영구 고착되는 것을 막는
        # 안전망이다 — 프레임이 계속 흐르는 동안만 검사된다.
        audio_end_at_s: float | None = None
        received_frames = 0
        committed_turns = 0
        audio_session_id: str | None = None
        session_started_at_s = time.perf_counter()

        async def safe_send(payload: Mapping[str, object]) -> bool:
            try:
                async with send_lock:
                    await websocket.send_json(dict(payload))
            except (RuntimeError, OSError, WebSocketDisconnect):
                return False
            return True

        async def safe_send_bytes(data: bytes) -> bool:
            try:
                async with send_lock:
                    await websocket.send_bytes(data)
            except (RuntimeError, OSError, WebSocketDisconnect):
                return False
            return True

        async def finish_turn() -> None:
            # Reopen the mic: clear the response gate and any partial detector
            # state accumulated before it engaged, then tell the client.
            nonlocal responding, last_reply, audio_end_at_s
            responding = False
            # The reply is off the speaker now: barge-ins after this point are
            # not echoes of it, so the echo check must disarm. Leaving it set
            # would compare THINKING-stage barge-ins of later turns against a
            # reply that finished playing long ago and eat real questions.
            last_reply = None
            audio_end_at_s = None
            if detector is not None:
                # The response window is over, so the anti-echo bar drops
                # with it (its other exit is the barge-in branch below).
                detector.echo_guard = False
                if not detector.speaking:
                    # Same principle as the playback_done guard: if the user
                    # has already started talking, resetting would erase their
                    # onset debounce and prefix — the first syllable dies.
                    detector.reset()
            await safe_send({"type": "listening"})

        async def respond(
            session_id: str,
            utterance: bytes,
            *,
            barged_in_turn: bool = False,
            previous_reply: str | None = None,
        ) -> None:
            """One voice turn: STT -> shared chat flow -> streamed TTS relay.

            Every blocking stage runs via asyncio.to_thread: a blocked event
            loop cannot flush socket writes, so without it the "streamed" TTS
            chunks would pile up in the transport buffer and arrive at the
            client as one lump when the pipeline finished.
            """
            nonlocal early_playback_done, last_reply, audio_end_at_s
            turn_started_at_s = time.perf_counter()

            def elapsed_ms() -> int:
                return round((time.perf_counter() - turn_started_at_s) * 1000)

            timings: dict[str, int] = {}
            # 32 bytes of PCM = 1 ms at 16 kHz int16 mono.
            await safe_send({"type": "turn", "duration_ms": len(utterance) // 32})
            try:
                client = await _speech_client()
                transcript = (await asyncio.to_thread(client.transcribe, utterance)).strip()
                timings["stt"] = elapsed_ms()
                if not transcript:
                    LOGGER.info("voice turn dropped: empty transcript")
                    await finish_turn()
                    return
                if len(transcript) > server_config.max_question_length:
                    transcript = transcript[: server_config.max_question_length]
                if barged_in_turn and previous_reply is not None:
                    overlap = _transcript_echo_overlap(transcript, previous_reply)
                    if overlap >= _ECHO_OVERLAP_THRESHOLD:
                        # The raised VAD bar let this "turn" through, but its
                        # words are mostly the reply that was playing: judge
                        # it as speaker echo, not the user, and skip the chat
                        # call entirely.
                        LOGGER.info(
                            "barge-in turn judged echo overlap=%.2f; dropped",
                            overlap,
                        )
                        await finish_turn()
                        return
                await safe_send({"type": "transcript", "text": transcript})
                LOGGER.info(
                    "voice turn transcribed chars=%s stt_ms=%s",
                    len(transcript),
                    timings["stt"],
                )

                status_code, payload = await answer_question(session_id, transcript)
                timings["llm"] = elapsed_ms() - timings["stt"]
                if status_code != 200:
                    await safe_send(payload)
                    await finish_turn()
                    return
                answer_text = str(payload.get("answer_text") or "")
                # Record before the relay, not after: if the user barges in
                # while this reply is on the speaker, the echo check of the
                # interrupting turn must compare against *this* text.
                last_reply = answer_text

                await safe_send({"type": "audio_start"})
                first_chunk_ms: int | None = None
                chunks = client.synthesize(answer_text)
                try:
                    # next() on the streaming response blocks too — same rule.
                    while (chunk := await asyncio.to_thread(next, chunks, None)) is not None:
                        if first_chunk_ms is None:
                            first_chunk_ms = elapsed_ms()
                            timings["tts_first"] = first_chunk_ms - timings["stt"] - timings["llm"]
                        if not await safe_send_bytes(chunk):
                            return
                finally:
                    closer = getattr(chunks, "close", None)
                    if callable(closer):
                        await asyncio.to_thread(closer)
                timings["tts_total"] = elapsed_ms() - timings["stt"] - timings["llm"]
                timings["total"] = elapsed_ms()
                audio_end_at_s = time.perf_counter()
                await safe_send(
                    {
                        "type": "audio_end",
                        "reply": answer_text,
                        "tool_calls": payload.get("tool_calls", []),
                        "timings": timings,
                    }
                )
                LOGGER.info(
                    "voice turn answered chars=%s timings=%s",
                    len(answer_text),
                    timings,
                )
                if early_playback_done:
                    # The client reported playback done (an early player
                    # error) before the relay finished; apply it now.
                    early_playback_done = False
                    await finish_turn()
            except asyncio.CancelledError:
                raise
            except SpeechServiceError as exc:
                LOGGER.warning("voice turn failed code=%s", exc.code)
                await safe_send(_error_payload(exc.code, exc.message))
                await finish_turn()
            except Exception:
                LOGGER.exception("voice turn failed unexpectedly")
                await safe_send(
                    _error_payload(
                        "VOICE_TURN_FAILED",
                        "voice turn could not be completed",
                    )
                )
                await finish_turn()

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
            parsed_session_id, start_error = _parse_audio_start_message(
                raw_start,
                server_config,
            )
            if parsed_session_id is None:
                await safe_send(
                    _error_payload("INVALID_START", start_error or "invalid start message")
                )
                await websocket.close(code=1008)
                return
            audio_session_id = parsed_session_id

            try:
                # 스레드로 미룬다: Silero 백엔드는 최초 로드에 torch import(초 단위)
                # + JIT 워밍업이 걸리는데, 이벤트 루프에서 하면 그동안 /ws/vision
                # 릴레이와 진행 중인 TTS 스트리밍까지 전부 멈춘다.
                detector = await asyncio.to_thread(build_turn_detector)
            except Exception:
                LOGGER.exception("turn detector initialization failed")
                await safe_send(
                    _error_payload(
                        "SESSION_INITIALIZATION_FAILED",
                        "audio session could not be initialized",
                    )
                )
                await websocket.close(code=1011)
                return

            # Same behavior as /ws/vision: a client-chosen session_id is
            # registered so voice-only use works without POST /api/session.
            store.register(audio_session_id)
            LOGGER.info(
                "audio session started session_id=%s silence_ms=%s prefix_ms=%s "
                "min_speech_ms=%s vad_aggressiveness=%s",
                audio_session_id,
                server_config.silence_ms,
                server_config.prefix_ms,
                server_config.min_speech_ms,
                server_config.vad_aggressiveness,
            )
            await safe_send({"type": "ready", "session_id": audio_session_id})

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_binary = message.get("bytes")
                raw_text = message.get("text")

                if raw_binary is not None:
                    received_frames += 1
                    if len(raw_binary) != FRAME_BYTES:
                        continue  # partial frame on connect/teardown; ignore
                    if (
                        responding
                        and audio_end_at_s is not None
                        and respond_task is not None
                        and respond_task.done()
                        and time.perf_counter() - audio_end_at_s > PLAYBACK_DONE_TIMEOUT_S
                    ):
                        # audio_end를 보낸 지 한참인데 playback_done이 없다 —
                        # 클라이언트가 재생을 끊고 mark를 잃은 경우다. 여기서
                        # 풀어 주지 않으면 responding+echo_guard가 영구 고착된다.
                        LOGGER.warning("playback_done timeout; releasing the turn")
                        await finish_turn()
                    if responding and not server_config.barge_in_enabled:
                        continue  # legacy mode: speech during a response is dropped
                    was_speaking = detector.speaking
                    utterance = detector.feed(raw_binary)
                    if detector.speaking != was_speaking:
                        if not await safe_send({"type": "vad", "speaking": detector.speaking}):
                            break
                    if responding and detector.speaking and not was_speaking:
                        # Barge-in: the user started talking over the agent.
                        # Cancel the relay now (its CancelledError path only
                        # re-raises, so no finish_turn/audio_end fires) and
                        # let the utterance keep accumulating toward a
                        # normal turn commit.
                        barged_in = True
                        responding = False
                        detector.echo_guard = False
                        if respond_task is not None:
                            respond_task.cancel()
                        if not await safe_send({"type": "interrupted"}):
                            break
                    elif barged_in and was_speaking and not detector.speaking and utterance is None:
                        # The interrupting utterance was discarded (below
                        # min_speech_ms): the next committed turn is a
                        # normal one again. The client saw "interrupted" and
                        # is waiting for the turn flow — close it explicitly
                        # so the contract's "interrupted is followed by the
                        # turn flow" holds even on this path.
                        barged_in = False
                        if not await safe_send({"type": "listening"}):
                            break
                    if utterance is not None:
                        responding = True
                        early_playback_done = False
                        committed_turns += 1
                        if server_config.barge_in_enabled:
                            # Raise the anti-echo bar for the whole response
                            # window; it drops again in finish_turn or the
                            # barge-in branch above.
                            detector.echo_guard = True
                        respond_task = asyncio.create_task(
                            respond(
                                audio_session_id,
                                utterance,
                                barged_in_turn=barged_in,
                                previous_reply=last_reply,
                            )
                        )
                        barged_in = False
                    continue

                if not isinstance(raw_text, str):
                    continue
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    if not await safe_send(
                        _error_payload("INVALID_MESSAGE", "control messages must be JSON")
                    ):
                        break
                    continue
                message_type = parsed.get("type") if isinstance(parsed, dict) else None
                if message_type == "playback_done":
                    # ≈ a telephony provider's "mark" event: the client's
                    # speaker is actually done, so the mic can reopen.
                    if detector is not None and detector.speaking:
                        # A barge-in is in progress: honoring the mark now
                        # would reset the detector and destroy the turn. The
                        # turn flow (turn -> transcript -> ...) supersedes it.
                        continue
                    if respond_task is not None and not respond_task.done():
                        early_playback_done = True
                        continue
                    await finish_turn()
                    continue
                if not await safe_send(
                    _error_payload(
                        "INVALID_MESSAGE",
                        f"unsupported message type: {message_type}",
                    )
                ):
                    break
        finally:
            cancellation: asyncio.CancelledError | None = None
            if respond_task is not None:
                respond_task.cancel()
                cancellation = await _wait_shielded(
                    respond_task,
                    "voice respond task failed during disconnect",
                )
                if respond_task.cancelled():
                    # The cancellation we requested ourselves, not the
                    # endpoint's own; do not re-raise it.
                    cancellation = None
            LOGGER.info(
                "audio session ended session_id=%s frames=%s turns=%s elapsed_s=%.3f",
                audio_session_id,
                received_frames,
                committed_turns,
                time.perf_counter() - session_started_at_s,
            )
            audio_gate.release()
            if cancellation is not None:
                raise cancellation

    demo_dir = Path(__file__).resolve().parent / "static"
    if demo_dir.is_dir():
        # Browser test terminal for /ws/audio at /demo/ (no client build needed).
        application.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

    return application


app = create_app()
