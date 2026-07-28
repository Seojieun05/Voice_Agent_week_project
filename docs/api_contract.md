# 서버 API 계약 (Android 앱용)

시각장애인 보조 시스템 서버의 클라이언트 계약 문서다. Android 앱과 테스트 클라이언트는 이 문서를 기준으로 구현한다.

- 기본 주소: `http://<server-host>:<port>` (WebSocket은 `ws://`)
- 모든 REST 요청/응답 본문은 JSON, UTF-8이다.
- 에러 응답 본문은 공통 형식을 따른다:

```json
{ "type": "error", "code": "SESSION_NOT_FOUND", "message": "설명 문자열" }
```

## 전체 흐름

```
앱 시작
 → POST /api/session          session_id 발급
 → WS /ws/vision (start에 같은 session_id 사용)
     카메라 JPEG 프레임 전송, analysis_events / narrations 수신
 → 사용자가 질문하면 (앱에서 STT 수행)
     POST /api/chat { session_id, user_question }
 → answer_text 수신 → Android TTS로 재생
```

`session_id`는 두 경로를 잇는 열쇠다. `/ws/vision`의 `start.session_id`와 `/api/chat`의 `session_id`가 같아야 서버가 최신 장면 분석 결과를 질문에 결합한다.

참고: `/ws/vision`의 `start`에 임의의 session_id를 보내도 서버가 그 세션을 자동 등록하므로, `/api/session` 없이 스트림만으로도 `/api/chat`을 쓸 수 있다. 다만 앱은 `/api/session`으로 발급받은 ID를 쓰는 것을 권장한다.

## GET /health

서버 상태 확인.

응답 `200`:

```json
{ "status": "ok", "active_session": false }
```

`active_session`은 현재 `/ws/vision` 스트림이 활성인지 여부다(서버는 동시 1개 스트림만 허용).

## POST /api/session

세션 발급. 요청 본문 없음.

응답 `200`:

```json
{ "session_id": "b81c9f52...", "created_at_ms": 1780000000000 }
```

## WebSocket /ws/vision

카메라 JPEG 프레임을 보내고 프레임별 분석 결과를 받는다. (기존 프로토콜 그대로 유지. 상세 검증 규칙은 `src/vision_agent/server.py` 참고.)

1. 연결 직후 텍스트 메시지로 start를 보낸다:

```json
{
  "type": "start",
  "session_id": "b81c9f52...",
  "source_width": 640,
  "source_height": 480,
  "source_fps": 15
}
```

2. 프레임마다 JSON 헤더 → JPEG 바이너리 순서로 보낸다:

```json
{ "type": "frame", "sequence_id": 1, "captured_at_ms": 1780000000000 }
```

3. 프레임별 응답 (`type: "analysis"`):

```json
{
  "type": "analysis",
  "sequence_id": 1,
  "captured_at_ms": 1780000000000,
  "server_received_at_ms": 1780000000100,
  "completed_at_ms": 1780000000180,
  "dropped_frames": 0,
  "received_frames": 1,
  "processed_frames": 1,
  "processing_fps": 12.5,
  "model_load_ms": 900.0,
  "analysis_events": [
    {
      "object_type": "pedestrian_signal",
      "event_type": "object_state_changed",
      "current_state": "GREEN",
      "confidence": 0.86
    }
  ],
  "narrations": ["보행자 신호가 초록색으로 바뀌었습니다."],
  "timings": { "queue_wait_ms": 1.0, "decode_ms": 2.0, "inference_ms": 40.0, "analysis_ms": 3.0, "total_server_ms": 60.0 }
}
```

서버는 프레임을 처리할 때마다 해당 세션의 장면 상태를 내부 저장소에 보관하며, 이것이 `/api/chat` 답변의 근거가 된다:

- `visible_objects`: 마지막 프레임에 실제로 보인 물체 요약(종류, 위치 왼쪽/중앙/오른쪽, 신호 상태, OCR로 읽은 글자 — 버스 번호·표지판·키오스크 문구). 매 프레임 교체되며 중요도 순(신호등 > 차량 > 사람 > 표지판·키오스크)으로 정렬된다. 신뢰도가 매우 낮은 감지는 제외되고, 애매한 감지는 `is_uncertain`으로만 표시된다.
- `analysis_events`/`narrations`: 최근 상태 변화 이벤트와 안내 문장. 개수 제한과 함께 최신 프레임 기준 TTL(이벤트 5초, 안내 8초)이 적용되어 카메라가 지나친 물체는 답변 근거에서 사라진다. 이벤트에는 `seconds_ago`(몇 초 전)가 붙는다.

서버는 질문을 자주 나오는 유형(진행 가능 여부, 횡단 가능 여부, 글자 읽기, 물체 위치, 장면 설명)으로 분류해 유형별 답변 템플릿을 적용한다. 예: "앞으로 가도 돼?"는 '네/아니요'로 시작하는 답을 받는다. VLM 프레임 선택은 최신 프레임 기준 1.5초 이내의 프레임만 후보로 사용해 과거 장면이 답변에 쓰이지 않게 한다.

주요 에러 코드: `SESSION_BUSY`, `INVALID_START`, `INVALID_FRAME_HEADER`, `INVALID_MESSAGE_ORDER`, `FRAME_TOO_LARGE`, `RATE_LIMITED`, `INVALID_JPEG`, `PROCESSING_FAILED`.

## POST /api/chat

사용자 질문 텍스트를 보내고 장면 기반 답변을 받는다. MVP에서는 앱이 마이크 입력을 STT로 변환한 뒤 텍스트를 보낸다.

요청:

```json
{ "session_id": "b81c9f52...", "user_question": "지금 건너도 돼?" }
```

- `user_question`: 공백 제외 1자 이상, 최대 500자(`VISION_SERVER_MAX_QUESTION_LENGTH`로 조정).

응답 `200`:

```json
{
  "type": "chat_answer",
  "session_id": "b81c9f52...",
  "answer_text": "보행자 신호가 초록색으로 감지되었습니다. 다만 주변 차량이나 실제 도로 상황은 직접 확인이 필요합니다.",
  "has_scene_analysis": true,
  "scene_state_updated_at_ms": 1780000000180,
  "vlm": { "used": false, "reason": null, "latency_ms": null }
}
```

- `answer_text`: 앱이 TTS로 읽어줄 한국어 문장.
- `has_scene_analysis`: 이 세션에서 분석된 프레임이 하나라도 있었는지. `false`면 장면 근거 없이 답한 것이다.
- `scene_state_updated_at_ms`: 답변에 사용된 장면 상태의 마지막 갱신 시각. 앱은 이 값이 오래됐으면(예: 5초 이상) 사용자에게 알릴 수 있다.
- `vlm`: Grok Vision fallback 메타데이터. YOLO 결과만으로 부족할 때(감지 없음·낮은 신뢰도·색/글자 등 시각 질문·상세 설명 요청) 서버가 최근 프레임 1장을 Grok Vision에 함께 보낸다. `used`가 true면 `reason`은 트리거 사유(`no_detections`/`low_confidence`/`question_needs_vision`/`detail_requested`), `latency_ms`는 VLM 호출 소요 시간. false면 `reason`에 미사용/실패 사유(`cooldown_active`, `no_recent_frame`, `vlm_failed:<code>` 등)가 담기며 답변은 YOLO 장면 정보만으로 생성된 것이다. VLM 실패는 HTTP 에러가 아니라 폴백으로 처리된다.

에러:

| status | code | 의미 |
| --- | --- | --- |
| 400 | `INVALID_SESSION_ID` | session_id가 비었거나 너무 길다 |
| 400 | `INVALID_QUESTION` | 질문이 비어 있다 |
| 400 | `QUESTION_TOO_LONG` | 질문이 최대 길이를 넘었다 |
| 404 | `SESSION_NOT_FOUND` | 서버가 모르는 session_id (`/api/session` 또는 `/ws/vision` start 필요) |
| 422 | (FastAPI 기본) | 필수 필드 누락 등 스키마 위반 |
| 502 | `UPSTREAM_ERROR` `UPSTREAM_TIMEOUT` `UPSTREAM_UNAVAILABLE` `INVALID_UPSTREAM_RESPONSE` `EMPTY_ANSWER` | Grok API 호출 실패 — 앱은 재시도하거나 최신 narration을 대신 읽어줄 수 있다 |
| 503 | `MISSING_API_KEY` | 서버에 Grok API 키가 설정되지 않았다 |
| 500 | `CHAT_FAILED` | 서버 내부 오류 |

Grok 호출이 실패해도 서버와 `/ws/vision` 스트림은 계속 동작한다.

## 서버 환경변수 (chat 관련)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GROK_API_KEY` (또는 `XAI_API_KEY`) | 없음(필수) | Grok API 키. 코드에 하드코딩하지 않는다 |
| `GROK_BASE_URL` | `https://api.x.ai/v1` | Grok API 베이스 URL |
| `GROK_MODEL` | `grok-4-fast-non-reasoning` | 텍스트 답변 모델 |
| `GROK_TIMEOUT_S` | `20` | 텍스트 호출 타임아웃(초) |
| `GROK_VISION_MODEL` | (`GROK_MODEL`과 동일) | VLM fallback 모델 |
| `GROK_VISION_TIMEOUT_S` | `30` | VLM 호출 타임아웃(초) |
| `GROK_VISION_MAX_RETRIES` | `1` | VLM 일시 오류(타임아웃·5xx) 재시도 횟수 |
| `VISION_SERVER_MAX_QUESTION_LENGTH` | `500` | 질문 최대 길이 |
| `VISION_SERVER_VLM_FALLBACK_ENABLED` | `true` | VLM fallback 사용 여부 |
| `VISION_SERVER_VLM_CONFIDENCE_THRESHOLD` | `0.45` | 이 값 미만이면 VLM 트리거 |
| `VISION_SERVER_VLM_COOLDOWN_S` | `5` | 세션별 VLM 호출 최소 간격(초) |
| `VISION_SERVER_VLM_MAX_IMAGE_BYTES` | `1048576` | VLM 전송 이미지 최대 크기 |
| `VISION_SERVER_VLM_MAX_IMAGE_DIM` | `1024` | VLM 전송 이미지 최대 변 길이(px) |
| `VISION_SERVER_FRAME_BUFFER_FRAMES` | `5` | 세션별 보관 프레임 수(메모리 전용) |
| `VISION_SERVER_FRAME_BUFFER_MAX_AGE_S` | `10` | VLM에 쓸 수 있는 프레임 최대 나이(초) |

수신 JPEG는 VLM 선택용 ring buffer(메모리)에만 잠시 보관되며 디스크나 로그에 저장되지 않는다. VLM 호출 시에도 로그에는 사용 여부·사유·지연시간만 남는다.

## 향후 확장 (아직 미구현)

- `WS /ws/audio`: 음성 chunk 업로드 → 서버 STT → `/api/chat`과 동일 흐름
- 서버측 TTS 음성 응답
- 인증/세션 토큰
