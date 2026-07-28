"""Grok-backed conversation layer for scene-grounded voice questions.

This module is intentionally separate from the vision pipeline: it receives an
already-serialized scene snapshot plus the user question, and produces one
short Korean answer. The API key is read from the environment and never
logged. Raw frames or audio never reach this module.
"""

from __future__ import annotations

import base64
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
    "scene_state 해석:\n"
    "- visible_objects: 지금 카메라에 실제로 보이는 물체 목록이며, 이것만이 현재 화면의 "
    "근거입니다. 중요도 순으로 정렬되어 있어 첫 번째 항목이 가장 중요한 물체입니다. "
    "각 항목에는 object_type(종류), position(왼쪽/중앙/오른쪽), state(신호 색 등 상태), "
    "text(물체에서 읽어낸 글자: 버스 번호, 표지판·화면 문구)가 들어 있습니다.\n"
    "- recent_events: 과거의 상태 변화 기록입니다(seconds_ago = 몇 초 전). "
    "visible_objects에 없는 물체는 이미 화면에서 사라진 것이므로 지금 보인다고 말하면 "
    "안 됩니다. 변화 이력을 물을 때만 참고합니다.\n"
    "- latest_narrations: 시스템이 최근 말한 안내 문장, seconds_since_last_frame: 마지막 "
    "프레임 이후 지난 시간(초)입니다.\n"
    "규칙:\n"
    "1. 한국어로 짧고 명확하게, 전체 3~4문장 이내로 답합니다.\n"
    "2. '앞에 뭐가 보여', '설명해줘' 같은 장면 설명 요청에는 다음 구조로 답합니다: "
    "먼저 한 문장으로 전체 장면을 요약하고, 이어서 visible_objects의 첫 번째(가장 중요한) "
    "물체 하나를 2문장 이내로 구체적으로 설명합니다(위치, 상태, 적힌 글자). 사소한 물체를 "
    "나열하지 않습니다.\n"
    "3. 물체에 text가 있으면 반드시 그 내용을 읽어줍니다. 예: 버스 번호, 표지판 문구, "
    "키오스크 화면 내용.\n"
    "4. 횡단보도, 신호등, 차량 접근 등 안전과 관련된 질문에는 '건너세요' 같은 직접적인 "
    "이동 명령을 하지 않습니다. 시스템이 감지한 근거를 말하고, 실제 주변 상황은 직접 "
    "확인이 필요하다는 점을 덧붙입니다.\n"
    "5. scene_state에 근거가 없으면 추측하지 말고, 관련 정보가 감지되지 않았다고 말합니다. "
    "seconds_since_last_frame이 5를 넘으면 정보가 오래되었을 수 있다고 덧붙입니다.\n"
    "6. 불확실성 표현은 is_uncertain이 true인 물체를 말할 때와 안전 관련 답변에만, "
    "'~로 보입니다', '~일 수 있습니다'처럼 자연스럽게 한 번만 사용합니다. '신뢰도가 "
    "낮습니다', '불확실한 상태입니다' 같은 기계적 문구를 반복하지 말고, confidence 수치나 "
    "stable_id 같은 내부 필드명은 답변에서 언급하지 않습니다."
)


VISION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\n추가 규칙 (이미지 제공 시):\n"
    "7. 첨부된 이미지는 사용자의 카메라가 방금 찍은 현재 장면입니다. 이미지를 직접 보고 "
    "답하되, scene_state의 YOLO 감지 결과와 교차 확인합니다. 이미지와 감지 결과가 다르면 "
    "이미지에서 직접 본 것을 우선합니다.\n"
    "8. 장면 설명 요청이면 같은 구조를 유지합니다: 한 문장 전체 요약 후, 가장 중요한 물체 "
    "하나를 2문장 이내로 구체적으로(위치, 상태, 적힌 글자·그림) 설명합니다.\n"
    "9. 이미지 속 글자나 표지판·화면 내용을 읽을 수 있으면 반드시 읽어줍니다."
)


class ChatServiceError(Exception):
    """Raised when an answer cannot be produced; carries a stable error code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ChatClientProtocol(Protocol):
    """Runtime contract used by the server and its test fakes."""

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str: ...

    def create_vision_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        jpeg_bytes: bytes,
    ) -> str: ...


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
    vision_model: str = ""
    vision_timeout_s: float = 30.0
    vision_max_retries: int = 1

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
        if self.vision_timeout_s <= 0.0:
            raise ValueError("vision_timeout_s must be positive")
        if self.vision_max_retries < 0:
            raise ValueError("vision_max_retries must not be negative")

    @property
    def resolved_vision_model(self) -> str:
        return self.vision_model.strip() or self.model

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
            vision_model=os.getenv("GROK_VISION_MODEL", "").strip(),
            vision_timeout_s=float(os.getenv("GROK_VISION_TIMEOUT_S", "30")),
            vision_max_retries=int(os.getenv("GROK_VISION_MAX_RETRIES", "1")),
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
        return self._complete(body, timeout_s=self._config.timeout_s)

    def create_vision_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        jpeg_bytes: bytes,
    ) -> str:
        """Answer with the current camera frame attached, retrying transient failures."""
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload_text = json.dumps(
            {"scene_state": dict(scene_state), "user_question": user_question},
            ensure_ascii=False,
        )
        body = {
            "model": self._config.resolved_vision_model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": payload_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        attempts = self._config.vision_max_retries + 1
        for attempt in range(attempts):
            try:
                return self._complete(body, timeout_s=self._config.vision_timeout_s)
            except ChatServiceError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                LOGGER.warning(
                    "Grok vision call failed (attempt %s/%s) code=%s; retrying",
                    attempt + 1,
                    attempts,
                    exc.code,
                )
        raise AssertionError("unreachable")  # pragma: no cover

    def _complete(self, body: Mapping[str, object], *, timeout_s: float) -> str:
        try:
            response = self._client.post("/chat/completions", json=dict(body), timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ChatServiceError(
                "UPSTREAM_TIMEOUT",
                "Grok API request timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatServiceError(
                "UPSTREAM_UNAVAILABLE",
                "Grok API request failed",
                retryable=True,
            ) from exc

        if response.status_code != 200:
            # Do not log or forward the upstream body: it may echo request
            # details, and clients only need a stable code.
            LOGGER.warning("Grok API returned status %s", response.status_code)
            raise ChatServiceError(
                "UPSTREAM_ERROR",
                f"Grok API returned status {response.status_code}",
                retryable=response.status_code >= 500,
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
