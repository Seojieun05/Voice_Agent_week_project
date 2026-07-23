from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vision_agent.server import ServerConfig, _default_session_factory, create_app


@dataclass(frozen=True, slots=True)
class _FakeEvent:
    sequence_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": "TEST_EVENT",
            "sequence_id": self.sequence_id,
        }


@dataclass(slots=True)
class _FakeAnalysis:
    sequence_id: int
    analysis_events: list[_FakeEvent] = field(init=False)
    narrations: list[str] = field(init=False)
    timings: dict[str, float] = field(
        default_factory=lambda: {"inference_ms": 4.5, "analysis_ms": 1.25}
    )

    def __post_init__(self) -> None:
        self.analysis_events = [_FakeEvent(self.sequence_id)]
        self.narrations = [f"frame {self.sequence_id}"]


class _FakeSession:
    model_load_ms = 12.5

    def __init__(
        self,
        *,
        block_first_frame: bool = False,
        created: threading.Event | None = None,
    ) -> None:
        self.block_first_frame = block_first_frame
        self.first_frame_started = threading.Event()
        self.release_first_frame = threading.Event()
        self.reset_completed = threading.Event()
        self.contexts: list[object] = []
        self.frames: list[np.ndarray] = []
        self.reset_calls = 0
        if created is not None:
            created.set()

    def process_frame(self, frame: np.ndarray, context: object) -> _FakeAnalysis:
        self.frames.append(frame.copy())
        self.contexts.append(context)
        if self.block_first_frame and len(self.contexts) == 1:
            self.first_frame_started.set()
            if not self.release_first_frame.wait(timeout=5.0):
                raise RuntimeError("test did not release the first frame")
        return _FakeAnalysis(getattr(context, "source_sequence_id"))

    def reset(self) -> None:
        self.reset_calls += 1
        self.reset_completed.set()


def _jpeg(value: int = 127, *, width: int = 24, height: int = 16) -> bytes:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _start(websocket: object, *, session_id: str = "test-session", source_fps: float = 15) -> None:
    websocket.send_json(
        {
            "type": "start",
            "session_id": session_id,
            "source_width": 24,
            "source_height": 16,
            "source_fps": source_fps,
        }
    )


def _send_frame(
    websocket: object,
    sequence_id: int,
    jpeg: bytes,
    *,
    captured_at_ms: int | None = 1_780_000_000_000,
) -> None:
    websocket.send_json(
        {
            "type": "frame",
            "sequence_id": sequence_id,
            "captured_at_ms": captured_at_ms,
        }
    )
    websocket.send_bytes(jpeg)


def _test_config(**overrides: object) -> ServerConfig:
    values: dict[str, object] = {
        "max_receive_fps": 0.0,
        "max_frame_bytes": 1024 * 1024,
        "max_frame_width": 128,
        "max_frame_height": 128,
    }
    values.update(overrides)
    return ServerConfig(**values)  # type: ignore[arg-type]


def test_health_is_available_without_loading_a_session() -> None:
    factory_calls = 0

    def factory() -> _FakeSession:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeSession()

    with TestClient(create_app(_test_config(), factory)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_session": False}
    assert factory_calls == 0


def test_default_factory_forwards_live_tracker_ocr_and_bus_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected_session = _FakeSession()

    def fake_create(config: object, **kwargs: object) -> _FakeSession:
        captured["config"] = config
        captured.update(kwargs)
        return expected_session

    monkeypatch.setattr("vision_agent.pipeline.create_vision_session", fake_create)
    config = _test_config(
        tracker="custom-botsort.yaml",
        kiosk_ocr_interval_frames=6,
        text_ocr_interval_frames=7,
        narrate_bus_approach=False,
    )

    result = _default_session_factory(config)()

    assert result is expected_session
    pipeline_config = captured["config"]
    assert getattr(pipeline_config, "source") == "<live>"
    assert getattr(pipeline_config, "tracker") == "custom-botsort.yaml"
    assert getattr(pipeline_config, "kiosk_ocr_interval_frames") == 6
    assert getattr(pipeline_config, "text_ocr_interval_frames") == 7
    assert getattr(pipeline_config, "narrate_bus_approach") is False
    assert captured["live_mode"] is True
    assert captured["tracker_override"] == "custom-botsort.yaml"
    assert captured["narrate_bus_approach"] is False


def test_analysis_response_contains_protocol_metrics_and_structured_output() -> None:
    session = _FakeSession()
    app = create_app(_test_config(), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket)
            _send_frame(websocket, 101, _jpeg())
            response = websocket.receive_json()
            websocket.close()

    assert session.reset_completed.wait(timeout=2.0)
    assert response["type"] == "analysis"
    assert response["sequence_id"] == 101
    assert response["captured_at_ms"] == 1_780_000_000_000
    assert response["server_received_at_ms"] <= response["completed_at_ms"]
    assert response["dropped_frames"] == 0
    assert response["received_frames"] == 1
    assert response["processed_frames"] == 1
    assert response["processing_fps"] == 0.0
    assert response["model_load_ms"] == pytest.approx(12.5)
    assert response["analysis_events"] == [{"event_type": "TEST_EVENT", "sequence_id": 101}]
    assert response["narrations"] == ["frame 101"]
    assert set(response["timings"]) >= {
        "queue_wait_ms",
        "decode_ms",
        "inference_ms",
        "analysis_ms",
        "total_server_ms",
    }
    assert session.frames[0].shape == (16, 24, 3)
    context = session.contexts[0]
    assert getattr(context, "source_sequence_id") == 101
    assert getattr(context, "processed_index") == 0
    assert getattr(context, "captured_at_s") == pytest.approx(1_780_000_000.0)
    assert session.reset_calls == 1


def test_latest_pending_frame_replaces_older_frames_and_counts_drops() -> None:
    session = _FakeSession(block_first_frame=True)
    app = create_app(_test_config(), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket)
            _send_frame(websocket, 1, _jpeg(10))
            assert session.first_frame_started.wait(timeout=2.0)

            _send_frame(websocket, 2, _jpeg(20))
            _send_frame(websocket, 3, _jpeg(30))
            _send_frame(websocket, 4, _jpeg(40))
            session.release_first_frame.set()

            responses = [websocket.receive_json(), websocket.receive_json()]

    assert [response["sequence_id"] for response in responses] == [1, 4]
    assert responses[-1]["dropped_frames"] == 2
    assert responses[-1]["received_frames"] == 4
    assert responses[-1]["processed_frames"] == 2
    assert [getattr(context, "source_sequence_id") for context in session.contexts] == [1, 4]
    assert [getattr(context, "processed_index") for context in session.contexts] == [0, 1]
    assert getattr(session.contexts[-1], "dropped_frames") == 2


def test_protocol_errors_and_invalid_jpeg_do_not_kill_connection() -> None:
    session = _FakeSession()
    app = create_app(_test_config(), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket)

            websocket.send_bytes(b"binary-before-header")
            assert websocket.receive_json()["code"] == "INVALID_MESSAGE_ORDER"

            websocket.send_json({"type": "frame", "sequence_id": 1, "captured_at_ms": None})
            websocket.send_json({"type": "frame", "sequence_id": 2, "captured_at_ms": None})
            wrong_order = websocket.receive_json()
            assert wrong_order["code"] == "INVALID_MESSAGE_ORDER"
            assert wrong_order["sequence_id"] == 1

            websocket.send_bytes(_jpeg())
            recovered = websocket.receive_json()
            assert recovered["type"] == "analysis"
            assert recovered["sequence_id"] == 2

            _send_frame(websocket, 3, b"not-a-jpeg")
            invalid_jpeg = websocket.receive_json()
            assert invalid_jpeg["code"] == "INVALID_JPEG"
            assert invalid_jpeg["sequence_id"] == 3

            _send_frame(websocket, 4, _jpeg())
            final_response = websocket.receive_json()
            assert final_response["type"] == "analysis"
            assert final_response["sequence_id"] == 4

    assert [getattr(context, "processed_index") for context in session.contexts] == [0, 1]


def test_oversized_frame_is_rejected_without_initializing_decoder() -> None:
    session = _FakeSession()
    app = create_app(_test_config(max_frame_bytes=8), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket)
            _send_frame(websocket, 5, b"012345678")
            response = websocket.receive_json()

    assert response["code"] == "FRAME_TOO_LARGE"
    assert response["sequence_id"] == 5
    assert session.contexts == []


def test_debug_jpeg_is_saved_only_when_directory_is_explicit(tmp_path: Path) -> None:
    session = _FakeSession()
    jpeg = _jpeg(42)
    app = create_app(_test_config(debug_frame_dir=tmp_path), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket)
            _send_frame(websocket, 8, jpeg)
            assert websocket.receive_json()["sequence_id"] == 8

    saved = list(tmp_path.glob("frame_*_8.jpg"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == jpeg


def test_one_second_rate_limit_allows_a_burst_up_to_configured_count() -> None:
    session = _FakeSession()
    app = create_app(_test_config(max_receive_fps=2.0), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            _start(websocket, source_fps=2.0)
            _send_frame(websocket, 1, _jpeg(10))
            assert websocket.receive_json()["sequence_id"] == 1
            _send_frame(websocket, 2, _jpeg(20))
            assert websocket.receive_json()["sequence_id"] == 2
            _send_frame(websocket, 3, _jpeg(30))
            limited = websocket.receive_json()

    assert limited["code"] == "RATE_LIMITED"
    assert limited["sequence_id"] == 3
    assert [getattr(context, "source_sequence_id") for context in session.contexts] == [1, 2]


def test_invalid_start_returns_error_and_closes_connection() -> None:
    session = _FakeSession()
    app = create_app(_test_config(), lambda: session)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            websocket.send_text(json.dumps({"type": "frame", "sequence_id": 1}))
            response = websocket.receive_json()
            assert response["code"] == "INVALID_START"
            with pytest.raises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()

    assert disconnected.value.code == 1008
    assert session.contexts == []
    assert session.reset_calls == 0


def test_second_concurrent_session_is_rejected_and_first_remains_usable() -> None:
    sessions: list[_FakeSession] = []

    def factory() -> _FakeSession:
        session = _FakeSession()
        sessions.append(session)
        return session

    app = create_app(_test_config(), factory)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as first:
            _start(first, session_id="first")
            with client.websocket_connect("/ws/vision") as second:
                busy = second.receive_json()
                assert busy["code"] == "SESSION_BUSY"
                with pytest.raises(WebSocketDisconnect) as disconnected:
                    second.receive_json()
            assert disconnected.value.code == 1013

            _send_frame(first, 7, _jpeg())
            assert first.receive_json()["sequence_id"] == 7
            first.close()

    assert len(sessions) == 1
    assert sessions[0].reset_completed.wait(timeout=2.0)
    assert sessions[0].reset_calls == 1


def test_disconnect_resets_session_before_releasing_gate() -> None:
    sessions: list[_FakeSession] = []

    def factory() -> _FakeSession:
        session = _FakeSession()
        sessions.append(session)
        return session

    app = create_app(_test_config(), factory)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as first:
            _start(first, session_id="first")
            _send_frame(first, 1, _jpeg())
            assert first.receive_json()["sequence_id"] == 1
            first.close()

        assert sessions[0].reset_completed.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while client.get("/health").json()["active_session"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.get("/health").json()["active_session"] is False
        with client.websocket_connect("/ws/vision") as second:
            _start(second, session_id="second")
            _send_frame(second, 2, _jpeg())
            assert second.receive_json()["sequence_id"] == 2
            second.close()

    assert len(sessions) == 2
    assert sessions[1].reset_completed.wait(timeout=2.0)
    assert [session.reset_calls for session in sessions] == [1, 1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_frame_bytes": 0}, "max_frame_bytes"),
        ({"max_receive_fps": -1.0}, "max_receive_fps"),
        ({"confidence": 1.1}, "confidence"),
        ({"kiosk_ocr_interval_frames": 0}, "kiosk_ocr_interval_frames"),
        ({"text_ocr_interval_frames": 0}, "text_ocr_interval_frames"),
    ],
)
def test_server_config_rejects_invalid_limits(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ServerConfig(**kwargs)  # type: ignore[arg-type]
