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
import time
from collections.abc import Callable, Mapping
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
    "각 항목에는 object_type(종류), position(왼쪽/중앙/오른쪽/전방 전체), "
    "distance(가까움/중간/멀리 — 화면에서 차지하는 크기로 추정한 거리), state(신호 색 등 "
    "상태), text(물체에서 읽어낸 글자: 버스 번호, 표지판·화면 문구)가 들어 있습니다.\n"
    "- recent_events: 과거의 상태 변화 기록입니다(seconds_ago = 몇 초 전). "
    "visible_objects에 없는 물체는 이미 화면에서 사라진 것이므로 지금 보인다고 말하면 "
    "안 됩니다. 변화 이력을 물을 때만 참고합니다.\n"
    "- latest_narrations: 시스템이 최근 말한 안내 문장, seconds_since_last_frame: 마지막 "
    "프레임 이후 지난 시간(초)입니다.\n"
    "규칙:\n"
    "1. 한국어로 짧고 명확하게, 전체 3~4문장 이내로 답합니다. 항상 지금 이 순간의 화면 "
    "기준으로 현재형으로 답하며, 몇 초 전 상황을 지금 상황처럼 말하지 않습니다.\n"
    "1-1. 답변은 음성으로 재생되며 사용자가 언제든 말을 걸어 재생을 끊을 수 있습니다. "
    "그래서 가장 중요한 것을 반드시 첫 문장에 넣습니다. 우선순위는 (1) 즉시 필요한 행동"
    "(멈추세요·기다리세요), (2) 위험 대상과 방향, (3) 예/아니요 같은 직접적인 답, "
    "(4) 나머지 설명 순입니다. 배경 설명이나 조건·전제를 앞에 두고 결론을 뒤에 놓지 "
    "않습니다. 첫 문장만 들어도 사용자가 무엇을 해야 하는지 알 수 있어야 합니다.\n"
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
    "9. 이미지 속 글자나 표지판·화면 내용을 읽을 수 있으면 반드시 읽어줍니다.\n"
    "10. 이미지 하단 가장자리에 보이는 사용자 자신의 발·다리·신발·지팡이는 사람이나 "
    "장애물로 취급하지 않고 언급하지 않습니다."
)


class ChatServiceError(Exception):
    """Raised when an answer cannot be produced; carries a stable error code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# Tool executor: (tool_name, parsed_arguments) -> JSON-serializable result.
ToolExecutor = Callable[[str, Mapping[str, object]], object]

CHAT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_scene",
            "description": (
                "현재 카메라에 보이는 모든 YOLO 감지 객체를 반환합니다. 각 객체에 클래스, "
                "confidence, bbox 좌표, 추적 ID, 방향(왼쪽/중앙/오른쪽), 거리(가까움/중간/"
                "멀리), 화면 점유 비율과 탐지 시각이 포함됩니다."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": (
                "특정 객체가 현재 화면에 보이는지 찾습니다. 탐지 여부, 화면 방향, 크기, "
                "confidence를 반환합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "찾을 객체 이름. 영문 클래스 이름(person, car, bus, "
                            "traffic_light, bollard 등) 또는 한국어 이름(사람, 버스, "
                            "신호등 등)"
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_traffic_light",
            "description": "신호등/보행자 신호의 상태와 판정 신뢰도, 마지막 갱신 시각을 반환합니다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": (
                "최근 몇 초 동안의 장면 변화를 반환합니다: 물체 등장/사라짐, 신호 변화 "
                "이벤트(몇 초 전인지 포함)와 시스템이 최근 말한 안내 문장. '방금 뭐가 "
                "지나갔어?', '뭐가 바뀌었어?' 같은 과거 질문에 사용하세요."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_detected_text",
            "description": (
                "OCR이 이미 확정한 화면 속 글자(버스 번호, 표지판 문구, 키오스크 화면 "
                "내용)를 즉시 반환합니다. 이미지 분석보다 빠르고 비용이 없으므로 글자 "
                "질문에는 이 도구를 먼저 시도하고, 비어 있을 때만 analyze_frame_with_vlm을 "
                "사용하세요."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_camera_status",
            "description": (
                "카메라 스트림 연결 여부, 마지막 분석 시각, 버퍼된 프레임 수 등 시스템 "
                "상태를 반환합니다. 화면 정보가 없거나 오래된 것 같을 때, 또는 사용자가 "
                "카메라가 잘 되는지 물을 때 사용하세요."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_frame_with_vlm",
            "description": (
                "현재 카메라 프레임 이미지를 비전 모델로 직접 분석합니다. 객체 목록만으로 "
                "답하기 어려울 때(장면 묘사, 글자·표지판 읽기, 길 상태 확인, 색상 등) "
                "사용하세요. 호출 간 쿨다운이 있어 실패할 수 있습니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "이미지에 대해 물어볼 구체적인 질문",
                    }
                },
                "required": ["question"],
            },
        },
    },
]

TOOL_PROMPT = (
    "도구 사용 규칙: 입력의 scene_state는 요약일 뿐이며, 화면에 대한 사실을 답하기 전에 "
    "반드시 도구로 최신 정보를 확인합니다.\n"
    "- 객체의 개수·목록·위치를 묻는 질문: get_current_scene을 호출합니다.\n"
    "- 특정 물체가 있는지 묻는 질문: find_object를 호출합니다.\n"
    "- 신호등·횡단 관련 질문: check_traffic_light를 호출합니다.\n"
    "- '방금', '아까', '바뀌었어' 같은 최근 변화 질문: get_recent_changes를 호출합니다.\n"
    "- 버스 번호·표지판·화면 글자 질문: read_detected_text를 먼저 호출하고, 비어 있으면 "
    "analyze_frame_with_vlm으로 이미지에서 직접 읽습니다.\n"
    "- 화면 정보가 없거나 오래됐거나 카메라 상태를 물으면: check_camera_status를 "
    "호출합니다.\n"
    "- 장면 묘사, 길·바닥 상태, 색상 등 이미지를 직접 봐야 하는 질문과 '가도 돼?' 같은 "
    "이동 판단: analyze_frame_with_vlm을 호출합니다.\n"
    "도구 결과에 없는 내용을 지어내지 않으며, 도구가 오류를 반환하면 가진 정보만으로 "
    "짧게 답합니다. 같은 도구를 반복 호출하지 않습니다."
)


class ChatClientProtocol(Protocol):
    """Runtime contract used by the server and its test fakes."""

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str: ...

    def create_vision_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        jpeg_bytes: bytes,
    ) -> str: ...

    def create_tool_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        execute_tool: ToolExecutor,
    ) -> tuple[str, list[dict[str, object]]]: ...


# Frequent question types get a dedicated answer template so users receive
# a direct answer first (e.g. 예/아니요) instead of a generic scene summary.
_QUESTION_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("crossing", ("건너", "횡단")),
    (
        "can_i_go",
        (
            "가도 돼",
            "가도돼",
            "가도 될",
            "가도될",
            "지나가도",
            "걸어가도",
            "출발해도",
            "이동해도",
            "앞으로 가",
        ),
    ),
    ("read_text", ("써있", "써 있", "쓰여", "적혀", "뭐라고", "뭐라", "글자", "글씨", "읽어")),
    ("find_object", ("어디", "찾아")),
    ("describe", ("보여", "보이", "설명", "묘사", "장면", "상황", "주변에 뭐")),
)

QUESTION_GUIDANCE: Mapping[str, str] = {
    "crossing": (
        "질문 유형: 횡단 가능 여부. 첫 문장에서 감지된 신호 상태를 바로 말합니다"
        "(예: '신호가 초록불로 감지됩니다'). '건너세요' 같은 직접 명령은 하지 말고, "
        "신호 상태와 근거를 말한 뒤 차량 등 실제 주변 확인이 필요하다고 한 번 덧붙입니다. "
        "초록불이라도 남은 시간이 보이고 10초 이내면 남은 시간을 알려주고 이번 신호에는 "
        "건너지 말고 다음 신호를 기다리라고 안내합니다. 빨간불이면 기다리라고 합니다. "
        "신호등이 감지되지 않으면 첫 문장에서 그렇게 말합니다."
    ),
    "can_i_go": (
        "질문 유형: 진행 가능 여부. 반드시 '네' 또는 '아니요'로 답을 시작합니다. "
        "기본 답은 '네'이고, 판단이 애매할 때도 '네'입니다. '아니요'는 다음 큰 문제가 있을 "
        "때만 씁니다: 바로 앞(distance '가까움'이면서 중앙/전방 전체)에 사람이나 장애물이 "
        "있음, 사람이 아주 많아 지나갈 틈이 없음, 차·오토바이·자전거·킥보드가 사용자 쪽으로 "
        "빠르게 다가오고 있음, 인도가 아니거나 진행할 길이 없음, 시야 대부분이 벽·문·기둥 "
        "같은 표면으로 막혀 있음. 이미지가 벽이나 큰 표면으로 가득 차 있으면 애매해도 "
        "'아니요. 바로 앞이 막혀 있어요.'라고 합니다. "
        "바로 앞에 사람이나 장애물이 있으면 비켜갈 방향을 안내할 수 있지만, 그 전에 그쪽 "
        "노면이 인도인지 확인합니다. 비어 있는 쪽이라는 이유만으로는 안내하지 않습니다"
        "(차도가 가장 비어 보이기 때문입니다). 차도·자전거도로·주차장이거나 노면을 확신할 "
        "수 없으면 방향을 제시하지 않고 '앞에 사람이 있어요. 잠시 멈추세요'처럼 정지만 "
        "안내합니다. 인도임이 분명할 때만 '앞에 사람이 있어요. 왼쪽으로 비켜가세요'처럼 "
        "방향을 함께 말합니다. "
        "다음은 절대 '아니요'의 근거가 아닙니다: 멀리 또는 중간 거리에 있는 사람(몇 명이든), "
        "왼쪽/오른쪽에만 있는 물체, 그리고 배경의 기차·기찻길·다리 위 차량·강 건너편이나 "
        "옆 차도의 차량처럼 사용자의 보행로 밖에 있는 것. 이런 것은 무시하거나 필요하면 "
        "'다만 ~에 있는 ~는 주의하세요'로만 짧게 언급합니다. "
        "이미지가 첨부되어 있으면 사용자가 걷는 길 자체가 비어 있는지를 기준으로 판단합니다. "
        "전방에 횡단보도가 있으면 보행자 신호가 초록불일 때만 건너도 된다고 안내합니다. "
        "초록불 남은 시간이 보이고 10초 이내면 남은 시간을 알려주고 이번 신호에는 건너지 "
        "말라고 안내하며, 빨간불이면 기다리라고 합니다."
    ),
    "read_text": (
        "질문 유형: 글자 읽기. 읽을 수 있는 글자 내용을 첫 문장에서 바로 말합니다. "
        "글자가 감지되지 않으면 첫 문장에서 그렇게 말합니다."
    ),
    "find_object": (
        "질문 유형: 물체 위치 찾기. 찾는 물체가 보이면 첫 문장에서 위치"
        "(왼쪽/중앙/오른쪽)를 바로 말합니다. 보이지 않으면 지금 화면에는 없다고 말합니다."
    ),
    "describe": (
        "질문 유형: 장면 설명. 한 문장으로 전체 장면을 요약한 뒤, 가장 중요한 물체 하나를 "
        "2문장 이내로 구체적으로(위치, 상태, 적힌 글자) 설명합니다."
    ),
}


def classify_question(user_question: str) -> str:
    """Classify a question into a frequent type, or 'general' if none match."""
    for question_type, keywords in _QUESTION_TYPE_KEYWORDS:
        if any(keyword in user_question for keyword in keywords):
            return question_type
    return "general"


def system_prompt_for(user_question: str, *, vision: bool = False) -> str:
    """Base prompt plus the answer template for the detected question type."""
    base = VISION_SYSTEM_PROMPT if vision else SYSTEM_PROMPT
    guidance = QUESTION_GUIDANCE.get(classify_question(user_question))
    if guidance is None:
        return base
    return f"{base}\n{guidance}"


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
        {"role": "system", "content": system_prompt_for(user_question)},
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
                {"role": "system", "content": system_prompt_for(user_question, vision=True)},
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

    def create_tool_answer(
        self,
        scene_state: Mapping[str, object],
        user_question: str,
        execute_tool: ToolExecutor,
        *,
        max_rounds: int = 4,
    ) -> tuple[str, list[dict[str, object]]]:
        """Agentic loop: let Grok call scene tools before answering.

        ``execute_tool`` runs each requested tool locally; its results are
        fed back as ``tool`` messages until Grok produces a final answer.
        Returns the answer plus a log of the tools that were called.
        """
        payload_text = json.dumps(
            {"scene_state": dict(scene_state), "user_question": user_question},
            ensure_ascii=False,
        )
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": system_prompt_for(user_question) + "\n" + TOOL_PROMPT,
            },
            {"role": "user", "content": payload_text},
        ]
        called: list[dict[str, object]] = []

        for _round in range(max_rounds):
            body = {
                "model": self._config.model,
                "messages": messages,
                "tools": CHAT_TOOLS,
                "tool_choice": "auto",
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            }
            message = self._request_message(body, timeout_s=self._config.timeout_s)
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                answer_text = str(message.get("content") or "").strip()
                if not answer_text:
                    raise ChatServiceError("EMPTY_ANSWER", "Grok API returned an empty answer")
                return answer_text, called

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                name = str((function or {}).get("name", "")).strip()
                raw_arguments = (function or {}).get("arguments")
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except (ValueError, TypeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                started_s = time.perf_counter()
                try:
                    result = execute_tool(name, arguments)
                except Exception:
                    # A broken tool must degrade the answer, not kill it.
                    LOGGER.exception("chat tool execution failed name=%s", name)
                    result = {"error": "tool_execution_failed"}
                called.append(
                    {
                        "name": name,
                        "latency_ms": round((time.perf_counter() - started_s) * 1000.0, 1),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") if isinstance(call, Mapping) else None,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        # Tool budget exhausted: demand a final answer without more calls.
        messages.append(
            {
                "role": "user",
                "content": "도구 호출 없이 지금까지 얻은 정보만으로 최종 답변을 말해주세요.",
            }
        )
        body = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        return self._complete(body, timeout_s=self._config.timeout_s), called

    def _request_message(self, body: Mapping[str, object], *, timeout_s: float) -> dict:
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
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatServiceError(
                "INVALID_UPSTREAM_RESPONSE",
                "Grok API returned an unexpected response shape",
            ) from exc
        if not isinstance(message, dict):
            raise ChatServiceError(
                "INVALID_UPSTREAM_RESPONSE",
                "Grok API returned an unexpected response shape",
            )
        return message

    def _complete(self, body: Mapping[str, object], *, timeout_s: float) -> str:
        message = self._request_message(body, timeout_s=timeout_s)
        try:
            answer = message["content"]
        except KeyError as exc:
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
