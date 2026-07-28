from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi.testclient import TestClient

import pytest

from vision_agent.chat import ChatServiceError
from vision_agent.scene_state import SceneSnapshot
from vision_agent.server import ServerConfig, _vlm_trigger_reason, create_app


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
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.analysis_events = [_FakeEvent(self.sequence_id)]
        self.narrations = [f"frame {self.sequence_id}"]


class _FakeSession:
    model_load_ms = 1.0

    def __init__(self) -> None:
        self.reset_completed = threading.Event()

    def process_frame(self, frame: np.ndarray, context: object) -> _FakeAnalysis:
        return _FakeAnalysis(getattr(context, "source_sequence_id"))

    def reset(self) -> None:
        self.reset_completed.set()


class _FakeChatClient:
    def __init__(self, answer: str = "테스트 답변입니다.") -> None:
        self.answer = answer
        self.calls: list[tuple[dict[str, object], str]] = []
        self.closed = False
        self.error: ChatServiceError | Exception | None = None
        self.vision_answer = "이미지 기반 답변입니다."
        self.vision_calls: list[tuple[dict[str, object], str, bytes]] = []
        self.vision_error: ChatServiceError | None = None

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str:
        self.calls.append((dict(scene_state), user_question))
        if self.error is not None:
            raise self.error
        return self.answer

    def create_vision_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        jpeg_bytes: bytes,
    ) -> str:
        self.vision_calls.append((dict(scene_state), user_question, jpeg_bytes))
        if self.vision_error is not None:
            raise self.vision_error
        return self.vision_answer

    def close(self) -> None:
        self.closed = True


class _TextOnlyChatClient:
    """Fake without vision support, mirroring pre-VLM injected clients."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str:
        self.calls.append(user_question)
        return "텍스트 전용 답변"


def _jpeg(value: int = 127) -> bytes:
    frame = np.full((16, 24, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _test_config(**overrides: object) -> ServerConfig:
    values: dict[str, object] = {
        "max_receive_fps": 0.0,
        "max_frame_bytes": 1024 * 1024,
        "max_frame_width": 128,
        "max_frame_height": 128,
    }
    values.update(overrides)
    return ServerConfig(**values)  # type: ignore[arg-type]


def _chat_app(chat_client, **config_overrides: object):
    return create_app(
        _test_config(**config_overrides),
        lambda: _FakeSession(),
        chat_client_factory=lambda: chat_client,
    )


def _stream_one_frame(client: TestClient, session_id: str) -> None:
    with client.websocket_connect("/ws/vision") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": session_id,
                "source_width": 24,
                "source_height": 16,
                "source_fps": 15,
            }
        )
        websocket.send_json({"type": "frame", "sequence_id": 1, "captured_at_ms": None})
        websocket.send_bytes(_jpeg())
        assert websocket.receive_json()["sequence_id"] == 1
        websocket.close()


def test_session_endpoint_issues_chat_ready_session() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        created = client.post("/api/session")
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert session_id
        assert created.json()["created_at_ms"] > 0

        answered = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "앞에 뭐가 보여?"},
        )

    assert answered.status_code == 200
    payload = answered.json()
    assert payload["type"] == "chat_answer"
    assert payload["session_id"] == session_id
    assert payload["answer_text"] == "테스트 답변입니다."
    assert payload["has_scene_analysis"] is False
    assert payload["scene_state_updated_at_ms"] is None
    scene_state, question = chat_client.calls[0]
    assert scene_state == {
        "visible_objects": [],
        "recent_events": [],
        "latest_narrations": [],
        "updated_at_ms": None,
    }
    assert question == "앞에 뭐가 보여?"


def test_chat_uses_latest_scene_state_from_vision_stream() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client, vlm_fallback_enabled=False)) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "session_id": "shared-session",
                    "source_width": 24,
                    "source_height": 16,
                    "source_fps": 15,
                }
            )
            for sequence_id in (1, 2):
                websocket.send_json(
                    {"type": "frame", "sequence_id": sequence_id, "captured_at_ms": None}
                )
                websocket.send_bytes(_jpeg())
                assert websocket.receive_json()["sequence_id"] == sequence_id

            response = client.post(
                "/api/chat",
                json={"session_id": "shared-session", "user_question": "지금 건너도 돼?"},
            )
            websocket.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_scene_analysis"] is True
    assert payload["scene_state_updated_at_ms"] > 0
    scene_state, question = chat_client.calls[0]
    assert question == "지금 건너도 돼?"
    assert scene_state["visible_objects"] == []
    assert scene_state["latest_narrations"] == ["frame 1", "frame 2"]
    assert [event["sequence_id"] for event in scene_state["recent_events"]] == [1, 2]


@dataclass(frozen=True, slots=True)
class _FakeDetection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _FakeResult:
    object_type: str
    state: str | None = None
    is_uncertain: bool = False
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _FakeRichAnalysis:
    detections: list[_FakeDetection]
    analysis_results_by_index: dict[int, _FakeResult]
    analysis_events: list[object] = field(default_factory=list)
    narrations: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class _FakeRichSession:
    """Session whose frames report currently visible objects (frame is 24px wide)."""

    model_load_ms = 1.0

    def process_frame(self, frame: np.ndarray, context: object) -> _FakeRichAnalysis:
        return _FakeRichAnalysis(
            detections=[
                _FakeDetection("bus", 0.9, (16.0, 0.0, 24.0, 10.0)),
                _FakeDetection("traffic light", 0.5, (0.0, 0.0, 8.0, 10.0)),
                _FakeDetection("person", 0.95, (8.0, 0.0, 16.0, 10.0)),
                _FakeDetection("person", 0.4, (0.0, 0.0, 8.0, 10.0)),
                _FakeDetection("chair", 0.2, (8.0, 0.0, 16.0, 10.0)),
                _FakeDetection("person", 0.9, (8.0, 0.0, 16.0, 10.0)),
                _FakeDetection("person", 0.13, (8.0, 0.0, 16.0, 10.0)),
            ],
            analysis_results_by_index={
                0: _FakeResult("bus", attributes={"route_number": "146"}),
                1: _FakeResult("pedestrian_signal", state="GREEN", is_uncertain=True),
                # Placeholder analyzer output (generic analyzer): the UNKNOWN
                # state and its uncertainty flag must not reach the chat AI,
                # and must not shield weak detections from the noise filter.
                5: _FakeResult("person", state="UNKNOWN", is_uncertain=True),
                6: _FakeResult("person", state="UNKNOWN", is_uncertain=True),
            },
        )

    def reset(self) -> None:
        pass


def test_chat_scene_state_includes_visible_objects_with_text_and_position() -> None:
    chat_client = _FakeChatClient()
    app = create_app(
        _test_config(),
        lambda: _FakeRichSession(),
        chat_client_factory=lambda: chat_client,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/vision") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "session_id": "rich-session",
                    "source_width": 24,
                    "source_height": 16,
                    "source_fps": 15,
                }
            )
            websocket.send_json({"type": "frame", "sequence_id": 1, "captured_at_ms": None})
            websocket.send_bytes(_jpeg())
            analysis_response = websocket.receive_json()
            assert analysis_response["sequence_id"] == 1

            response = client.post(
                "/api/chat",
                json={"session_id": "rich-session", "user_question": "앞에 뭐가 보여?"},
            )
            websocket.close()

    assert response.status_code == 200
    scene_state, _question = chat_client.calls[0]
    # Importance-ordered (signal > vehicle > person), no raw confidence
    # numbers, weak unconfirmed detections flagged or dropped entirely.
    assert scene_state["visible_objects"] == [
        {
            "object_type": "pedestrian_signal",
            "position": "왼쪽",
            "distance": "중간",
            "state": "GREEN",
            "is_uncertain": True,
        },
        {
            "object_type": "bus",
            "position": "오른쪽",
            "distance": "중간",
            "text": "146",
        },
        {
            "object_type": "person",
            "position": "중앙",
            "distance": "중간",
        },
        {
            "object_type": "person",
            "position": "중앙",
            "distance": "중간",
        },
        {
            "object_type": "person",
            "position": "왼쪽",
            "distance": "중간",
            "is_uncertain": True,
        },
    ]
    assert scene_state["seconds_since_last_frame"] >= 0
    # The WebSocket response format is unchanged by the chat feature.
    assert "visible_objects" not in analysis_response


def test_chat_rejects_unknown_session() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        response = client.post(
            "/api/chat",
            json={"session_id": "never-registered", "user_question": "질문"},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "SESSION_NOT_FOUND"
    assert chat_client.calls == []


def test_chat_validates_question_and_session_id() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        session_id = client.post("/api/session").json()["session_id"]

        blank_question = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "   "},
        )
        long_question = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "가" * 501},
        )
        blank_session = client.post(
            "/api/chat",
            json={"session_id": "  ", "user_question": "질문"},
        )
        missing_fields = client.post("/api/chat", json={"session_id": session_id})

    assert blank_question.status_code == 400
    assert blank_question.json()["code"] == "INVALID_QUESTION"
    assert long_question.status_code == 400
    assert long_question.json()["code"] == "QUESTION_TOO_LONG"
    assert blank_session.status_code == 400
    assert blank_session.json()["code"] == "INVALID_SESSION_ID"
    assert missing_fields.status_code == 422
    assert chat_client.calls == []


def test_chat_survives_upstream_failure_and_recovers() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        session_id = client.post("/api/session").json()["session_id"]

        chat_client.error = ChatServiceError("UPSTREAM_ERROR", "Grok API returned status 500")
        failed = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "질문"},
        )

        chat_client.error = None
        recovered = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "질문"},
        )

    assert failed.status_code == 502
    assert failed.json() == {
        "type": "error",
        "code": "UPSTREAM_ERROR",
        "message": "Grok API returned status 500",
    }
    assert recovered.status_code == 200
    assert recovered.json()["answer_text"] == "테스트 답변입니다."


def test_chat_reports_missing_api_key_as_service_unavailable() -> None:
    def failing_factory() -> _FakeChatClient:
        raise ChatServiceError("MISSING_API_KEY", "Grok API key is not configured")

    app = create_app(
        _test_config(),
        lambda: _FakeSession(),
        chat_client_factory=failing_factory,
    )
    with TestClient(app) as client:
        session_id = client.post("/api/session").json()["session_id"]
        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "질문"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "MISSING_API_KEY"


def test_chat_wraps_unexpected_errors_without_dying() -> None:
    chat_client = _FakeChatClient()
    chat_client.error = RuntimeError("boom")

    with TestClient(_chat_app(chat_client)) as client:
        session_id = client.post("/api/session").json()["session_id"]
        failed = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "질문"},
        )
        health = client.get("/health")

    assert failed.status_code == 500
    assert failed.json()["code"] == "CHAT_FAILED"
    assert health.status_code == 200


def test_chat_client_is_created_once_and_closed_on_shutdown() -> None:
    chat_client = _FakeChatClient()
    factory_calls = 0

    def factory() -> _FakeChatClient:
        nonlocal factory_calls
        factory_calls += 1
        return chat_client

    app = create_app(_test_config(), lambda: _FakeSession(), chat_client_factory=factory)
    with TestClient(app) as client:
        session_id = client.post("/api/session").json()["session_id"]
        for _ in range(2):
            response = client.post(
                "/api/chat",
                json={"session_id": session_id, "user_question": "질문"},
            )
            assert response.status_code == 200

    assert factory_calls == 1
    assert chat_client.closed is True


def _snapshot(
    visible_objects: tuple[dict[str, object], ...],
    scene_confidence: float | None,
) -> SceneSnapshot:
    return SceneSnapshot(
        session_id="s",
        visible_objects=visible_objects,
        recent_events=(),
        latest_narrations=(),
        updated_at_ms=1,
        scene_confidence=scene_confidence,
    )


@pytest.mark.parametrize(
    ("visible_objects", "scene_confidence", "question", "expected"),
    [
        ((), None, "앞에 뭐가 보여?", "no_detections"),
        (({"object_type": "person"},), 0.3, "앞에 뭐가 보여?", "low_confidence"),
        (({"object_type": "bus"},), 0.9, "버스가 무슨 색이야?", "question_needs_vision"),
        (({"object_type": "sign"},), 0.9, "표지판에 뭐라고 적혀 있어?", "question_needs_vision"),
        (({"object_type": "person"},), 0.9, "자세히 설명해줘", "detail_requested"),
        (({"object_type": "person"},), 0.9, "앞으로 가도 돼?", "path_check"),
        (({"object_type": "person"},), 0.9, "지금 건너도 돼?", "path_check"),
        (({"object_type": "person"},), 0.9, "앞에 뭐가 보여?", None),
        (
            ({"object_type": "person", "distance": "멀리"},),
            0.9,
            "앞에 뭐가 보여?",
            "sparse_scene",
        ),
        (
            ({"object_type": "person", "distance": "중간"},),
            0.9,
            "앞에 뭐가 보여?",
            None,
        ),
        (
            ({"object_type": "sign", "distance": "멀리", "text": "출구"},),
            0.9,
            "앞에 뭐가 보여?",
            None,
        ),
    ],
)
def test_vlm_trigger_reasons(
    visible_objects: tuple[dict[str, object], ...],
    scene_confidence: float | None,
    question: str,
    expected: str | None,
) -> None:
    config = _test_config()

    assert (
        _vlm_trigger_reason(_snapshot(visible_objects, scene_confidence), question, config)
        == expected
    )


def test_vlm_trigger_disabled_by_config() -> None:
    config = _test_config(vlm_fallback_enabled=False)

    assert _vlm_trigger_reason(_snapshot((), None), "앞에 뭐가 보여?", config) is None


def test_vlm_used_when_no_detections_and_frame_available() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        _stream_one_frame(client, "vlm-session")
        response = client.post(
            "/api/chat",
            json={"session_id": "vlm-session", "user_question": "앞에 뭐가 보여?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_text"] == "이미지 기반 답변입니다."
    assert payload["vlm"]["used"] is True
    assert payload["vlm"]["reason"] == "no_detections"
    assert payload["vlm"]["latency_ms"] >= 0
    assert chat_client.calls == []
    scene_state, question, jpeg_bytes = chat_client.vision_calls[0]
    assert question == "앞에 뭐가 보여?"
    assert "visible_objects" in scene_state
    assert jpeg_bytes.startswith(b"\xff\xd8")


def test_vlm_failure_falls_back_to_text_answer() -> None:
    chat_client = _FakeChatClient()
    chat_client.vision_error = ChatServiceError("UPSTREAM_TIMEOUT", "timed out")

    with TestClient(_chat_app(chat_client)) as client:
        _stream_one_frame(client, "vlm-session")
        response = client.post(
            "/api/chat",
            json={"session_id": "vlm-session", "user_question": "앞에 뭐가 보여?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_text"] == "테스트 답변입니다."
    assert payload["vlm"]["used"] is False
    assert payload["vlm"]["reason"] == "vlm_failed:UPSTREAM_TIMEOUT"
    assert len(chat_client.vision_calls) == 1
    assert len(chat_client.calls) == 1


def test_vlm_cooldown_forces_text_path_on_second_question() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client, vlm_cooldown_s=60.0)) as client:
        _stream_one_frame(client, "vlm-session")
        first = client.post(
            "/api/chat",
            json={"session_id": "vlm-session", "user_question": "앞에 뭐가 보여?"},
        )
        second = client.post(
            "/api/chat",
            json={"session_id": "vlm-session", "user_question": "앞에 뭐가 보여?"},
        )

    assert first.json()["vlm"]["used"] is True
    payload = second.json()
    assert payload["vlm"]["used"] is False
    assert payload["vlm"]["reason"] == "cooldown_active"
    assert payload["answer_text"] == "테스트 답변입니다."
    assert len(chat_client.vision_calls) == 1


def test_vlm_skipped_without_recent_frame() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        session_id = client.post("/api/session").json()["session_id"]
        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "앞에 뭐가 보여?"},
        )

    payload = response.json()
    assert payload["vlm"]["used"] is False
    assert payload["vlm"]["reason"] == "no_recent_frame"
    assert payload["answer_text"] == "테스트 답변입니다."
    assert chat_client.vision_calls == []


def test_vlm_keyword_trigger_with_rich_scene() -> None:
    chat_client = _FakeChatClient()
    app = create_app(
        _test_config(),
        lambda: _FakeRichSession(),
        chat_client_factory=lambda: chat_client,
    )

    with TestClient(app) as client:
        _stream_one_frame(client, "rich-vlm")
        response = client.post(
            "/api/chat",
            json={"session_id": "rich-vlm", "user_question": "버스에 뭐라고 써있어?"},
        )

    payload = response.json()
    assert payload["vlm"]["used"] is True
    assert payload["vlm"]["reason"] == "question_needs_vision"


def test_text_only_chat_client_still_works_with_vlm_enabled() -> None:
    chat_client = _TextOnlyChatClient()

    with TestClient(_chat_app(chat_client)) as client:
        _stream_one_frame(client, "legacy-session")
        response = client.post(
            "/api/chat",
            json={"session_id": "legacy-session", "user_question": "앞에 뭐가 보여?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_text"] == "텍스트 전용 답변"
    assert payload["vlm"]["used"] is False
    assert payload["vlm"]["reason"] == "client_unsupported"


def test_position_label_uses_overlap_not_center() -> None:
    from vision_agent.server import _position_label

    # Object occupying the right half: center is near the middle but the
    # user experiences it on the right.
    assert _position_label(50.0, 100.0, 100) == "오른쪽"
    assert _position_label(0.0, 50.0, 100) == "왼쪽"
    assert _position_label(40.0, 60.0, 100) == "중앙"
    # Very wide object blocks the whole path.
    assert _position_label(10.0, 90.0, 100) == "전방 전체"
    assert _position_label(0.0, 0.0, 100) is None


def test_vlm_frame_selection_never_picks_stale_frames() -> None:
    from vision_agent.frame_buffer import BufferedFrame
    from vision_agent.server import _select_vlm_image

    def noisy_jpeg() -> bytes:
        rng = np.random.default_rng(seed=7)
        frame = rng.integers(0, 255, size=(16, 24, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        return encoded.tobytes()

    config = _test_config()
    old_sharp = BufferedFrame(sequence_id=1, received_at_ms=0, jpeg_bytes=noisy_jpeg())
    new_flat = BufferedFrame(sequence_id=2, received_at_ms=10_000, jpeg_bytes=_jpeg(200))

    selected = _select_vlm_image([old_sharp, new_flat], config)

    assert selected is not None
    decoded = cv2.imdecode(np.frombuffer(selected, dtype=np.uint8), cv2.IMREAD_COLOR)
    # The stale-but-sharp frame is outside the freshness window, so the
    # flat bright frame must win.
    assert float(decoded.mean()) > 150.0
    assert float(decoded.std()) < 30.0


def test_vlm_frame_selection_prefers_sharper_frame_within_window() -> None:
    from vision_agent.frame_buffer import BufferedFrame
    from vision_agent.server import _select_vlm_image

    rng = np.random.default_rng(seed=11)
    noisy = rng.integers(0, 255, size=(16, 24, 3), dtype=np.uint8)
    ok, noisy_encoded = cv2.imencode(".jpg", noisy)
    assert ok

    config = _test_config()
    sharp = BufferedFrame(sequence_id=1, received_at_ms=9_800, jpeg_bytes=noisy_encoded.tobytes())
    flat = BufferedFrame(sequence_id=2, received_at_ms=10_000, jpeg_bytes=_jpeg(200))

    selected = _select_vlm_image([sharp, flat], config)

    assert selected is not None
    decoded = cv2.imdecode(np.frombuffer(selected, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Both frames are fresh; the noisy (higher Laplacian variance) one wins.
    assert float(decoded.std()) > 30.0


def test_distance_label_scales_with_apparent_size() -> None:
    from vision_agent.server import _distance_label

    # Far pedestrian: small box relative to the frame.
    assert _distance_label(45.0, 55.0, 40.0, 55.0, 100, 100) == "멀리"
    # Mid-distance: about a third of the frame height.
    assert _distance_label(40.0, 60.0, 30.0, 65.0, 100, 100) == "중간"
    # A person filling ~60% of the height but floating mid-frame is a few
    # meters away — walking past them is fine, so not "가까움".
    assert _distance_label(30.0, 50.0, 10.0, 70.0, 100, 100) == "중간"
    # Same size but reaching the frame bottom: right in front of the user.
    assert _distance_label(30.0, 60.0, 30.0, 100.0, 100, 100) == "가까움"
    # Dominant box filling most of the frame height.
    assert _distance_label(20.0, 80.0, 10.0, 90.0, 100, 100) == "가까움"
    assert _distance_label(0.0, 10.0, 0.0, 10.0, 0, 100) is None


def test_wide_background_object_not_reported_as_blocking() -> None:
    from vision_agent.server import _serialized_visible_objects

    # A train spanning the whole frame width in the background (short box
    # high in the frame) must not read as "전방 전체".
    analysis = _FakeRichAnalysis(
        detections=[_FakeDetection("train", 0.8, (0.0, 4.0, 24.0, 7.0))],
        analysis_results_by_index={},
    )

    objects, _confidence, _raw = _serialized_visible_objects(analysis, 24, 16)

    assert objects == [{"object_type": "train", "position": "중앙", "distance": "중간"}]


def test_wide_close_object_keeps_blocking_position() -> None:
    from vision_agent.server import _serialized_visible_objects

    analysis = _FakeRichAnalysis(
        detections=[_FakeDetection("bench", 0.8, (0.0, 2.0, 24.0, 16.0))],
        analysis_results_by_index={},
    )

    objects, _confidence, _raw = _serialized_visible_objects(analysis, 24, 16)

    assert objects == [{"object_type": "bench", "position": "전방 전체", "distance": "가까움"}]


def test_own_feet_person_box_is_ignored() -> None:
    from vision_agent.server import _serialized_visible_objects

    # Frame 24x16. A "person" starting low and running off the bottom edge
    # is the user's own feet/legs; a real close person shows a head higher
    # in the frame and must be kept.
    analysis = _FakeRichAnalysis(
        detections=[
            _FakeDetection("person", 0.9, (8.0, 10.0, 16.0, 16.0)),
            _FakeDetection("person", 0.9, (8.0, 2.0, 16.0, 16.0)),
        ],
        analysis_results_by_index={},
    )

    objects, _confidence, _raw = _serialized_visible_objects(analysis, 24, 16)

    assert objects == [{"object_type": "person", "position": "중앙", "distance": "가까움"}]


def _raw_snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        session_id="s",
        visible_objects=(
            {
                "object_type": "pedestrian_signal",
                "position": "중앙",
                "state": "GREEN",
                "is_uncertain": True,
            },
        ),
        recent_events=(
            {"object_type": "pedestrian_signal", "event_type": "changed", "seconds_ago": 2},
            {"object_type": "bus", "event_type": "appeared", "seconds_ago": 1},
        ),
        latest_narrations=(),
        updated_at_ms=1,
        scene_confidence=0.9,
        raw_detections=(
            {
                "class_name": "traffic light",
                "confidence": 0.62,
                "bbox_xyxy": [1.0, 1.0, 5.0, 8.0],
                "track_id": 3,
                "direction": "중앙",
                "distance": "멀리",
                "size_percent": 1.2,
            },
            {
                "class_name": "bus",
                "confidence": 0.9,
                "bbox_xyxy": [10.0, 2.0, 20.0, 12.0],
                "track_id": 7,
                "direction": "오른쪽",
                "distance": "중간",
                "size_percent": 20.8,
            },
        ),
    )


def test_tool_get_current_scene_returns_raw_detections() -> None:
    from vision_agent.server import _tool_get_current_scene

    result = _tool_get_current_scene(_raw_snapshot())

    assert result["object_count"] == 2
    assert result["detected_at_ms"] == 1
    assert result["seconds_since_detection"] >= 0
    bus = result["detected_objects"][1]
    assert bus["class_name"] == "bus"
    assert bus["bbox_xyxy"] == [10.0, 2.0, 20.0, 12.0]
    assert bus["track_id"] == 7


def test_tool_find_object_matches_korean_alias() -> None:
    from vision_agent.server import _tool_find_object

    found = _tool_find_object(_raw_snapshot(), "버스")
    assert found["found"] is True
    assert found["matches"][0]["class_name"] == "bus"
    assert found["matches"][0]["direction"] == "오른쪽"
    assert found["matches"][0]["size_percent"] == 20.8
    assert found["matches"][0]["confidence"] == 0.9

    missing = _tool_find_object(_raw_snapshot(), "kiosk")
    assert missing["found"] is False
    assert missing["matches"] == []

    invalid = _tool_find_object(_raw_snapshot(), "   ")
    assert invalid == {"error": "name_required"}


def test_tool_check_traffic_light_reports_state_and_confidence() -> None:
    from vision_agent.server import _tool_check_traffic_light

    result = _tool_check_traffic_light(_raw_snapshot())

    assert result["traffic_light_visible"] is True
    assert result["state"] == "GREEN"
    assert result["is_uncertain"] is True
    assert result["detection_confidence"] == 0.62
    assert result["last_update_ms"] == 1
    assert [event["object_type"] for event in result["recent_signal_events"]] == [
        "pedestrian_signal"
    ]


def test_tool_check_traffic_light_when_no_signal() -> None:
    from vision_agent.server import _tool_check_traffic_light

    empty = SceneSnapshot(
        session_id="s",
        visible_objects=(),
        recent_events=(),
        latest_narrations=(),
        updated_at_ms=None,
    )

    result = _tool_check_traffic_light(empty)

    assert result["traffic_light_visible"] is False
    assert result["state"] == "UNKNOWN"
    assert result["detection_confidence"] is None


class _FakeToolChatClient(_FakeChatClient):
    """Fake that exercises every server tool through the executor."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_results: dict[str, object] = {}

    def create_tool_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        execute_tool,
    ) -> tuple[str, list[dict[str, object]]]:
        self.calls.append((dict(scene_state), user_question))
        self.tool_results = {
            "scene": execute_tool("get_current_scene", {}),
            "find": execute_tool("find_object", {"name": "버스"}),
            "light": execute_tool("check_traffic_light", {}),
            "vlm": execute_tool("analyze_frame_with_vlm", {"question": "길 상태는?"}),
            "unknown": execute_tool("bogus_tool", {}),
        }
        return "툴 기반 답변입니다.", [
            {"name": "get_current_scene", "latency_ms": 1.0},
            {"name": "analyze_frame_with_vlm", "latency_ms": 2.0},
        ]


def test_chat_prefers_tool_calling_path_and_executes_server_tools() -> None:
    chat_client = _FakeToolChatClient()
    app = create_app(
        _test_config(),
        lambda: _FakeRichSession(),
        chat_client_factory=lambda: chat_client,
    )

    with TestClient(app) as client:
        _stream_one_frame(client, "tool-session")
        response = client.post(
            "/api/chat",
            json={"session_id": "tool-session", "user_question": "버스 어디 있어?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_text"] == "툴 기반 답변입니다."
    assert [call["name"] for call in payload["tool_calls"]] == [
        "get_current_scene",
        "analyze_frame_with_vlm",
    ]
    # analyze_frame_with_vlm ran through the ring buffer + vision client.
    assert payload["vlm"]["used"] is True
    assert payload["vlm"]["reason"] == "tool_call"
    assert chat_client.tool_results["vlm"] == {"vlm_answer": "이미지 기반 답변입니다."}
    assert len(chat_client.vision_calls) == 1

    scene = chat_client.tool_results["scene"]
    assert scene["object_count"] == 7  # raw keeps every detection
    find = chat_client.tool_results["find"]
    assert find["found"] is True
    assert find["matches"][0]["class_name"] == "bus"
    light = chat_client.tool_results["light"]
    assert light["state"] == "GREEN"
    assert chat_client.tool_results["unknown"] == {"error": "unknown_tool:bogus_tool"}


def test_vlm_tool_respects_cooldown_and_missing_frames() -> None:
    chat_client = _FakeToolChatClient()
    app = create_app(
        _test_config(vlm_cooldown_s=60.0),
        lambda: _FakeRichSession(),
        chat_client_factory=lambda: chat_client,
    )

    with TestClient(app) as client:
        # No frames streamed yet: session known via /api/session only.
        session_id = client.post("/api/session").json()["session_id"]
        client.post(
            "/api/chat",
            json={"session_id": session_id, "user_question": "질문"},
        )
        assert chat_client.tool_results["vlm"] == {"error": "no_recent_frame"}

        _stream_one_frame(client, "cooldown-session")
        client.post(
            "/api/chat",
            json={"session_id": "cooldown-session", "user_question": "질문"},
        )
        assert chat_client.tool_results["vlm"] == {"vlm_answer": "이미지 기반 답변입니다."}

        client.post(
            "/api/chat",
            json={"session_id": "cooldown-session", "user_question": "질문"},
        )
        assert chat_client.tool_results["vlm"]["error"] == "cooldown_active"
        assert chat_client.tool_results["vlm"]["retry_after_s"] > 0
