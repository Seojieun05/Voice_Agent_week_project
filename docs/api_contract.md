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

핸즈프리 대안 — 앱(또는 브라우저)이 STT/TTS 없이 원시 오디오만 스트리밍:

```
 → WS /ws/audio (start에 같은 session_id 사용)
     마이크 PCM 프레임 전송 → 서버가 턴 종료 판단(VAD+endpointing)
     → 서버 STT → chat 흐름 → TTS 오디오 청크 수신 → 재생
     → 재생 완료 시 playback_done 전송
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

- `visible_objects`: 마지막 프레임에 실제로 보인 물체 요약(종류, 위치 왼쪽/중앙/오른쪽/전방 전체, 거리 가까움/중간/멀리, 신호 상태, OCR로 읽은 글자 — 버스 번호·표지판·키오스크 문구). 매 프레임 교체되며 중요도 순(신호등 > 차량 > 사람 > 표지판·키오스크)으로 정렬된다. 신뢰도가 매우 낮은 감지는 제외되고, 애매한 감지는 `is_uncertain`으로만 표시된다.
- `analysis_events`/`narrations`: 최근 상태 변화 이벤트와 안내 문장. 개수 제한과 함께 최신 프레임 기준 TTL(이벤트 5초, 안내 8초)이 적용되어 카메라가 지나친 물체는 답변 근거에서 사라진다. 이벤트에는 `seconds_ago`(몇 초 전)가 붙는다.

## 접근 위험 이벤트 (`hazard_detected`)

추적된 박스가 커지는 속도로 충돌 예상 시간(TTC)을 추정해, 사용자에게 빠르게 다가오는 물체를 알린다. 박스 면적은 거리의 제곱에 반비례하므로 `TTC = 2 / (d(ln 면적)/dt)`로 구한다 — 카메라 보정이나 물체 실제 크기가 필요 없다.

기존 `object_approaching`과는 별개다. 그쪽은 버스 분석기가 정류장 진입 같은 느린 움직임을 판정하는 것이고, 이 이벤트는 클래스와 무관하게 박스 변화만 본다.

```json
{
  "object_type": "bicycle",
  "event_type": "hazard_detected",
  "current_state": "IMMINENT",
  "confidence": 0.82,
  "attributes": {
    "hazard_level": "IMMINENT",
    "zone": "CENTER",
    "time_to_contact_s": 1.1,
    "in_path": true,
    "emission_index": 1
  }
}
```

- `hazard_level`: `IMMINENT`(기본 TTC 1.5초 이내) 또는 `WARNING`(3.5초 이내).
- `WARNING`은 스스로 움직이는 물체(사람·자전거·킥보드·오토바이·차량·버스·트럭 등)가 사용자 진행 방향에 있을 때만 나간다. 볼라드·기둥 같은 정적 장애물은 `IMMINENT`에서만 알린다 — 걸어가다 부딪히기 직전이라는 뜻이다.
- `zone`: `LEFT`/`CENTER`/`RIGHT`. `in_path`는 진행 통로(화면 가운데 50%)와 겹치는지.
- 안내 문장은 항상 행동을 먼저 말한다: `"위험, 멈추세요. 정면에서 자전거가 빠르게 다가옵니다."` 사용자가 끼어들어 재생을 끊어도 첫 문장으로 필요한 정보가 전달되게 한 것이다. 이 이벤트는 narration 우선순위 0으로 신호 변화보다 먼저 나가고, TTL은 1.5초로 짧다(지난 "멈추세요"는 방해가 된다).

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
  "vlm": { "used": false, "reason": null, "latency_ms": null },
  "tool_calls": [ { "name": "find_object", "latency_ms": 1.2 } ]
}
```

- `answer_text`: 앱이 TTS로 읽어줄 한국어 문장.
- `has_scene_analysis`: 이 세션에서 분석된 프레임이 하나라도 있었는지. `false`면 장면 근거 없이 답한 것이다.
- `scene_state_updated_at_ms`: 답변에 사용된 장면 상태의 마지막 갱신 시각. 앱은 이 값이 오래됐으면(예: 5초 이상) 사용자에게 알릴 수 있다.
- `tool_calls`: 답변 생성 중 Grok이 호출한 서버 도구 목록. 서버는 Grok에 7개 도구를 제공하며 Grok이 질문에 따라 자율적으로 호출한다:
  - `get_current_scene()` — 최신 YOLO 감지 전체(클래스, confidence, bbox, 추적 ID, 방향, 거리, 화면 점유율, 탐지 시각)
  - `find_object(name)` — 특정 객체 탐지 여부·방향·크기·confidence (영문 클래스명 또는 한국어 별칭: 사람/버스/신호등/볼라드 등)
  - `check_traffic_light()` — 신호 상태, 판정 신뢰도, 마지막 갱신 시각, 최근 신호 이벤트
  - `get_recent_changes()` — 최근 몇 초간의 등장/사라짐/상태 변화 이벤트(seconds_ago)와 최근 안내 문장
  - `read_detected_text()` — OCR이 확정한 화면 속 글자(버스 번호·표지판·키오스크 문구)를 VLM 호출 없이 즉시 반환
  - `check_camera_status()` — 스트림 연결 여부, 마지막 분석 시각, 버퍼된 프레임 수 등 데이터 신선도 진단
  - `analyze_frame_with_vlm(question)` — 최근 프레임 1장을 Grok Vision으로 직접 분석 (cooldown 적용, 실패 시 오류 객체 반환)
- `vlm`: Grok Vision fallback 메타데이터. YOLO 결과만으로 부족할 때(감지 없음·낮은 신뢰도·색/글자 등 시각 질문·상세 설명 요청·진행/횡단 판단) 서버가 최근 프레임 1장을 Grok Vision에 함께 보낸다. `used`가 true면 `reason`은 트리거 사유(`no_detections`/`low_confidence`/`question_needs_vision`/`detail_requested`/`path_check`), `latency_ms`는 VLM 호출 소요 시간. false면 `reason`에 미사용/실패 사유(`cooldown_active`, `no_recent_frame`, `vlm_failed:<code>` 등)가 담기며 답변은 YOLO 장면 정보만으로 생성된 것이다. VLM 실패는 HTTP 에러가 아니라 폴백으로 처리된다.

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

## WebSocket /ws/audio

핸즈프리 음성 대화. 클라이언트가 마이크 오디오를 연속 스트리밍하면 **서버가
말이 끝나는 시점을 스스로 판단**(VAD + endpointing)해 STT → `/api/chat`과 동일한
답변 흐름 → TTS를 수행하고, TTS 오디오를 청크 단위로 되돌려준다. 버튼이나
푸시투톡이 필요 없다.

오디오 계약(고정): **mono int16 little-endian, 16 kHz, 20 ms 프레임(640바이트)**.
제어 메시지는 JSON 텍스트, 오디오는 바이너리 메시지다. 동시 오디오 세션은
1개만 허용된다(`/ws/vision`의 세션 제한과는 별개).

1. 연결 직후 텍스트 메시지로 start를 보낸다 (`/ws/vision`과 같은 session_id를
   쓰면 최신 장면 분석이 답변에 결합된다):

```json
{ "type": "start", "session_id": "b81c9f52..." }
```

서버 응답: `{ "type": "ready", "session_id": "..." }`. 알 수 없는 session_id는
`/ws/vision`과 동일하게 자동 등록된다(장면 정보 없이 동작).

2. 이후 클라이언트는 640바이트 PCM 프레임을 바이너리로 계속 보낸다(~50/초).
   크기가 틀린 프레임은 조용히 무시된다.

3. 서버 → 클라이언트 메시지 (한 턴의 순서):

| 메시지 | 의미 |
| --- | --- |
| `{ "type": "vad", "speaking": bool }` | 음성 감지 상태 변화 (UI 표시용) |
| `{ "type": "turn", "duration_ms": int }` | 턴 종료 판정 — 발화 길이(ms) |
| `{ "type": "transcript", "text": "..." }` | STT 결과 |
| `{ "type": "audio_start" }` | TTS 스트리밍 시작 — 클라이언트는 마이크를 닫는다 |
| (바이너리) | TTS mp3 청크, 도착 즉시 재생 가능 |
| `{ "type": "audio_end", "reply": "...", "tool_calls": [...], "timings": {...} }` | 답변 텍스트·도구 목록·단계별 지연(ms: stt/llm/tts_first/tts_total/total) |
| `{ "type": "interrupted" }` | 끼어들기(barge-in) 확정 — 서버가 응답을 취소했다. 아래 5 참조 |
| `{ "type": "listening" }` | 마이크를 다시 열어도 됨 |

4. 재생이 **실제로 끝나면** 클라이언트는 `{ "type": "playback_done" }`을 보낸다
   (전화망의 mark 이벤트에 해당). 서버는 `listening`으로 응답해 마이크 재개를
   확정한다. 단, 감지기가 사용자 발화 중(barge-in 진행 중)이라고 판단하는 동안
   도착한 `playback_done`은 완전히 무시된다 — 진행 중인 턴이 리셋되면 안 되기
   때문이다.

5. **barge-in(끼어들기, 기본 활성)**: 클라이언트는 응답 재생 중에도 PCM
   프레임을 계속 보낸다. 서버는 응답 중 수신 프레임을 버리지 않고 **에코
   가드**(VAD 문턱을 `VISION_SERVER_VAD_GUARD_THRESHOLD`로 상향)를 올린 채
   같은 턴 감지기에 계속 먹인다. 그래도 사용자 발화 시작이 확인되는 순간
   서버는 `{ "type": "interrupted" }`를 1회 보내고 진행 중이던 응답을
   취소한다(TTS 릴레이 중단, `audio_end` 없음). 클라이언트는 `interrupted`
   수신 즉시 재생을 멈추고 버퍼를 버려야 한다. 이후 발화는 일반 턴 흐름
   (`turn` → `transcript` → 새 응답)으로 이어진다. 2차 에코 방어로, 끼어든
   턴의 STT 결과가 직전 답변 텍스트와 단어 집합 기준 60% 이상 겹치면 스피커
   에코로 판정해 chat 호출 없이 `listening`만 보낸다.
   `VISION_SERVER_BARGE_IN=0`이면 기존 동작(응답 중 수신 오디오 폐기)으로
   되돌아가며 `interrupted`는 발생하지 않는다.

6. 턴 처리 실패(STT/Grok/TTS)는 `error` 메시지 + `listening`으로 통지되며 연결은
   유지된다. 주요 코드: `STT_TIMEOUT`/`STT_ERROR`/`STT_UNAVAILABLE`,
   `TTS_TIMEOUT`/`TTS_ERROR`/`TTS_UNAVAILABLE`, `/api/chat`과 동일한 chat 계열
   코드, `VOICE_TURN_FAILED`. 연결 수준 오류: `INVALID_START`(close 1008),
   `SESSION_BUSY`(close 1013), `SESSION_INITIALIZATION_FAILED`(close 1011).

브라우저 테스트 페이지가 `http://<server>:<port>/demo/`에 있다 (Chrome/Edge,
마이크 필요). 서버 실행 후 페이지를 열고 그냥 말하면 전체 흐름을 확인할 수 있다.
발화 오디오는 STT 호출에만 사용되고 디스크나 로그에 저장되지 않는다.

## 서버 환경변수 (음성 관련)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `TTS_VOICE` | `ara` | xAI TTS 목소리 (ara/eve/leo/rex/sal) |
| `TTS_LANGUAGE` | `auto` | TTS 언어 힌트 |
| `GROK_STT_TIMEOUT_S` | `20` | STT 호출 타임아웃(초) |
| `GROK_TTS_TIMEOUT_S` | `30` | TTS 호출 타임아웃(초) |
| `VISION_SERVER_SILENCE_MS` | `700` | 이만큼 연속 무음이면 턴 종료 ("patience dial" — 낮추면 빠르지만 말을 끊고, 높이면 안 끊지만 응답이 늦다) |
| `VISION_SERVER_PREFIX_MS` | `300` | 발화 시작 전 보존할 오디오 (첫 음절 잘림 방지) |
| `VISION_SERVER_MIN_SPEECH_MS` | `250` | 실제 음성이 이보다 짧으면 버림 (기침·소음이 API 호출로 이어지는 것 방지) |
| `VISION_SERVER_VAD_AGGRESSIVENESS` | `2` | webrtcvad 민감도 0(관대)~3(엄격) |
| `VISION_SERVER_VAD_BACKEND` | `silero` | VAD 백엔드 `silero`/`webrtc` — silero 로드 실패 시 webrtc로 자동 폴백 |
| `VISION_SERVER_VAD_THRESHOLD` | `0.5` | silero 발화 판정 확률 문턱 (0~1] |
| `VISION_SERVER_VAD_GUARD_THRESHOLD` | `0.75` | 에코 가드(TTS 재생 중) 상향 문턱 — threshold 이상이어야 한다 |
| `VISION_SERVER_MAX_UTTERANCE_MS` | `30000` | 발화 최대 길이 — 초과 시 강제 턴 종료 |
| `VISION_SERVER_BARGE_IN` | `1` | 끼어들기(barge-in) 활성 여부 — `0`/`false`만 비활성(응답 중 수신 오디오 폐기로 복귀) |

STT/TTS는 chat과 같은 `GROK_API_KEY`(또는 `XAI_API_KEY`)와 `GROK_BASE_URL`을 쓴다.

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

- `/ws/audio` 대화 기억(멀티턴 컨텍스트)
- 인증/세션 토큰
