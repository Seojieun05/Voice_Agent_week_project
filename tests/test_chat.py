from __future__ import annotations

import json

import httpx
import pytest

from vision_agent.chat import (
    DEFAULT_GROK_MODEL,
    SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    ChatServiceError,
    GrokChatClient,
    GrokConfig,
    build_chat_messages,
)

SCENE_STATE = {
    "recent_events": [
        {
            "object_type": "pedestrian_signal",
            "event_type": "object_state_changed",
            "current_state": "GREEN",
            "confidence": 0.86,
        }
    ],
    "latest_narrations": ["보행자 신호가 초록색으로 바뀌었습니다."],
    "updated_at_ms": 1_780_000_000_000,
}


def _client(handler, **config_overrides) -> GrokChatClient:
    config = GrokConfig(api_key="test-key", base_url="https://grok.test/v1", **config_overrides)
    return GrokChatClient(config, transport=httpx.MockTransport(handler))


def _answer_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": text}}]},
    )


def test_build_chat_messages_embeds_scene_state_and_question() -> None:
    messages = build_chat_messages(SCENE_STATE, "지금 건너도 돼?")

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    payload = json.loads(messages[1]["content"])
    assert payload["scene_state"] == SCENE_STATE
    assert payload["user_question"] == "지금 건너도 돼?"


def test_create_answer_sends_expected_request_and_returns_text() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return _answer_response(
            "  보행자 신호가 초록색으로 감지되었습니다. 주변 상황은 직접 확인이 필요합니다.  "
        )

    answer = _client(handler).create_answer(SCENE_STATE, "지금 건너도 돼?")

    assert answer == "보행자 신호가 초록색으로 감지되었습니다. 주변 상황은 직접 확인이 필요합니다."
    assert captured["url"] == "https://grok.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == DEFAULT_GROK_MODEL
    assert body["messages"] == build_chat_messages(SCENE_STATE, "지금 건너도 돼?")


def test_create_answer_maps_upstream_status_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler).create_answer(SCENE_STATE, "question")

    assert excinfo.value.code == "UPSTREAM_ERROR"


def test_create_answer_maps_timeouts() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler).create_answer(SCENE_STATE, "question")

    assert excinfo.value.code == "UPSTREAM_TIMEOUT"


def test_create_answer_maps_transport_failures() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler).create_answer(SCENE_STATE, "question")

    assert excinfo.value.code == "UPSTREAM_UNAVAILABLE"


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"unexpected": True},
    ],
)
def test_create_answer_rejects_malformed_upstream_payloads(payload: dict[str, object]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler).create_answer(SCENE_STATE, "question")

    assert excinfo.value.code == "INVALID_UPSTREAM_RESPONSE"


def test_create_answer_rejects_empty_answers() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _answer_response("   ")

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler).create_answer(SCENE_STATE, "question")

    assert excinfo.value.code == "EMPTY_ANSWER"


def test_create_vision_answer_attaches_image_and_uses_vision_model() -> None:
    captured: dict[str, object] = {}
    jpeg = b"\xff\xd8fake-jpeg-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _answer_response("빨간색 문이 보입니다.")

    client = _client(handler, vision_model="grok-vision-test")
    answer = client.create_vision_answer(SCENE_STATE, "문이 무슨 색이야?", jpeg)

    assert answer == "빨간색 문이 보입니다."
    body = captured["body"]
    assert body["model"] == "grok-vision-test"
    assert body["messages"][0] == {"role": "system", "content": VISION_SYSTEM_PROMPT}
    content = body["messages"][1]["content"]
    text_part = content[0]
    payload = json.loads(text_part["text"])
    assert payload["scene_state"] == SCENE_STATE
    assert payload["user_question"] == "문이 무슨 색이야?"
    image_part = content[1]
    expected_b64 = __import__("base64").b64encode(jpeg).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


def test_create_vision_answer_falls_back_to_text_model_name() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _answer_response("답변")

    _client(handler).create_vision_answer(SCENE_STATE, "질문", b"jpeg")

    assert captured["body"]["model"] == DEFAULT_GROK_MODEL


def test_create_vision_answer_retries_transient_failures_once() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "overloaded"})
        return _answer_response("복구된 답변")

    answer = _client(handler, vision_max_retries=1).create_vision_answer(
        SCENE_STATE,
        "질문",
        b"jpeg",
    )

    assert answer == "복구된 답변"
    assert len(attempts) == 2


def test_create_vision_answer_does_not_retry_client_errors() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler, vision_max_retries=2).create_vision_answer(SCENE_STATE, "질문", b"jpeg")

    assert excinfo.value.code == "UPSTREAM_ERROR"
    assert len(attempts) == 1


def test_create_vision_answer_raises_after_retries_exhausted() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(ChatServiceError) as excinfo:
        _client(handler, vision_max_retries=1).create_vision_answer(SCENE_STATE, "질문", b"jpeg")

    assert excinfo.value.code == "UPSTREAM_TIMEOUT"
    assert len(attempts) == 2


def test_config_from_environment_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(ChatServiceError) as excinfo:
        GrokConfig.from_environment()

    assert excinfo.value.code == "MISSING_API_KEY"


def test_config_from_environment_reads_key_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "env-key")
    monkeypatch.setenv("GROK_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("GROK_MODEL", "grok-custom")
    monkeypatch.setenv("GROK_TIMEOUT_S", "5")
    monkeypatch.setenv("GROK_VISION_MODEL", "grok-vision-custom")
    monkeypatch.setenv("GROK_VISION_TIMEOUT_S", "12")
    monkeypatch.setenv("GROK_VISION_MAX_RETRIES", "2")

    config = GrokConfig.from_environment()

    assert config.api_key == "env-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "grok-custom"
    assert config.timeout_s == 5.0
    assert config.resolved_vision_model == "grok-vision-custom"
    assert config.vision_timeout_s == 12.0
    assert config.vision_max_retries == 2


def test_config_from_environment_accepts_xai_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GROK_BASE_URL", raising=False)
    monkeypatch.delenv("GROK_MODEL", raising=False)
    monkeypatch.delenv("GROK_TIMEOUT_S", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-key")

    config = GrokConfig.from_environment()

    assert config.api_key == "xai-key"
    assert config.model == DEFAULT_GROK_MODEL


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"api_key": "  "}, "api_key"),
        ({"api_key": "k", "model": " "}, "model"),
        ({"api_key": "k", "timeout_s": 0.0}, "timeout_s"),
        ({"api_key": "k", "max_tokens": 0}, "max_tokens"),
        ({"api_key": "k", "vision_timeout_s": 0.0}, "vision_timeout_s"),
        ({"api_key": "k", "vision_max_retries": -1}, "vision_max_retries"),
    ],
)
def test_grok_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GrokConfig(**kwargs)  # type: ignore[arg-type]
