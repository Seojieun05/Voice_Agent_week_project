from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi.testclient import TestClient

from vision_agent.chat import ChatServiceError
from vision_agent.server import ServerConfig, create_app


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

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str:
        self.calls.append((dict(scene_state), user_question))
        if self.error is not None:
            raise self.error
        return self.answer

    def close(self) -> None:
        self.closed = True


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


def _chat_app(chat_client: _FakeChatClient):
    return create_app(
        _test_config(),
        lambda: _FakeSession(),
        chat_client_factory=lambda: chat_client,
    )


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
    assert scene_state == {"recent_events": [], "latest_narrations": [], "updated_at_ms": None}
    assert question == "앞에 뭐가 보여?"


def test_chat_uses_latest_scene_state_from_vision_stream() -> None:
    chat_client = _FakeChatClient()

    with TestClient(_chat_app(chat_client)) as client:
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
    assert scene_state["latest_narrations"] == ["frame 1", "frame 2"]
    assert [event["sequence_id"] for event in scene_state["recent_events"]] == [1, 2]


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
