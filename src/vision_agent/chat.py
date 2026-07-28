"""Grok-backed conversation layer for scene-grounded voice questions.

This module is intentionally separate from the vision pipeline: it receives an
already-serialized scene snapshot plus the user question, and produces one
short Korean answer. The API key is read from the environment and never
logged. Raw frames or audio never reach this module.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

LOGGER = logging.getLogger(__name__)

GROK_API_KEY_ENVS = ("GROK_API_KEY", "XAI_API_KEY")
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-4-fast-non-reasoning"

SYSTEM_PROMPT = (
    "당신은 시각장애인을 위한 야외 보행 보조 시스템의 음성 안내 AI입니다. "
    "카메라 영상 분석 결과(scene_state)와 사용자 질문이 JSON으로 주어집니다.\n"
    "규칙:\n"
    "1. 한국어로 한두 문장, 짧고 명확하게 답합니다.\n"
    "2. 횡단보도, 신호등, 차량 접근 등 안전과 관련된 질문에는 '건너세요' 같은 직접적인 "
    "이동 명령을 하지 않습니다. 시스템이 감지한 근거를 말하고, 실제 주변 상황은 직접 "
    "확인이 필요하다는 불확실성을 함께 전달합니다.\n"
    "3. scene_state에 근거가 없으면 추측하지 말고, 관련 정보가 감지되지 않았다고 말합니다.\n"
    "4. confidence가 낮은 감지 결과는 단정하지 않고 가능성으로 표현합니다."
)


class ChatServiceError(Exception):
    """Raised when an answer cannot be produced; carries a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ChatClientProtocol(Protocol):
    """Runtime contract used by the server and its test fakes."""

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str: ...


def build_chat_messages(
    scene_state: Mapping[str, object],
    user_question: str,
) -> list[dict[str, str]]:
    """Build the structured Grok messages from a scene snapshot and question."""
    payload = {
        "scene_state": dict(scene_state),
        "user_question": user_question,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


@dataclass(frozen=True, slots=True)
class GrokConfig:
    api_key: str
    base_url: str = DEFAULT_GROK_BASE_URL
    model: str = DEFAULT_GROK_MODEL
    timeout_s: float = 20.0
    temperature: float = 0.2
    max_tokens: int = 300

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    @classmethod
    def from_environment(cls) -> GrokConfig:
        api_key = ""
        for name in GROK_API_KEY_ENVS:
            raw_value = os.getenv(name)
            if raw_value is not None and raw_value.strip():
                api_key = raw_value.strip()
                break
        if not api_key:
            raise ChatServiceError(
                "MISSING_API_KEY",
                "Grok API key is not configured; set GROK_API_KEY or XAI_API_KEY",
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("GROK_BASE_URL", DEFAULT_GROK_BASE_URL).strip()
            or DEFAULT_GROK_BASE_URL,
            model=os.getenv("GROK_MODEL", DEFAULT_GROK_MODEL).strip() or DEFAULT_GROK_MODEL,
            timeout_s=float(os.getenv("GROK_TIMEOUT_S", "20")),
        )


class GrokChatClient:
    """Synchronous Grok chat-completions client.

    The blocking HTTP call is intended to run off the event loop (for example
    via ``asyncio.to_thread``). A custom ``transport`` can be injected so tests
    never reach the network.
    """

    def __init__(
        self,
        config: GrokConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    @classmethod
    def from_environment(cls) -> GrokChatClient:
        return cls(GrokConfig.from_environment())

    def close(self) -> None:
        self._client.close()

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str:
        body = {
            "model": self._config.model,
            "messages": build_chat_messages(scene_state, user_question),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        try:
            response = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ChatServiceError(
                "UPSTREAM_TIMEOUT",
                "Grok API request timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatServiceError(
                "UPSTREAM_UNAVAILABLE",
                "Grok API request failed",
            ) from exc

        if response.status_code != 200:
            # Do not log or forward the upstream body: it may echo request
            # details, and clients only need a stable code.
            LOGGER.warning("Grok API returned status %s", response.status_code)
            raise ChatServiceError(
                "UPSTREAM_ERROR",
                f"Grok API returned status {response.status_code}",
            )

        try:
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatServiceError(
                "INVALID_UPSTREAM_RESPONSE",
                "Grok API returned an unexpected response shape",
            ) from exc

        answer_text = str(answer).strip()
        if not answer_text:
            raise ChatServiceError(
                "EMPTY_ANSWER",
                "Grok API returned an empty answer",
            )
        return answer_text
