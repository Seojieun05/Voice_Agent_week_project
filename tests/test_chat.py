from __future__ import annotations

import json

import httpx
import pytest

from vision_agent.chat import (
    DEFAULT_GROK_MODEL,
    SYSTEM_PROMPT,
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


def _client(handler) -> GrokChatClient:
    config = GrokConfig(api_key="test-key", base_url="https://grok.test/v1")
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

    config = GrokConfig.from_environment()

    assert config.api_key == "env-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "grok-custom"
    assert config.timeout_s == 5.0


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
    ],
)
def test_grok_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GrokConfig(**kwargs)  # type: ignore[arg-type]
