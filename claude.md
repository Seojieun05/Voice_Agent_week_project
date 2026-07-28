프로젝트 개요
이 프로젝트는 시각장애인을 위한 야외용 스마트 글래스/카메라 보조 시스템의 서버 레포지토리다.
최종 구조는 다음을 목표로 한다.
XIAO ESP32S3 Sense 또는 스마트폰 카메라
→ 서버로 실시간 JPEG 프레임 전송
→ YOLO 기반 장면 분석
→ 사용자 음성 질문 수신
→ Grok API 기반 AI 응답 생성
→ 텍스트 또는 TTS 결과를 Android 앱으로 반환
→ Galaxy Buds/유선 이어폰으로 출력
Android 앱은 별도 레포지토리에서 구현한다. 이 레포에서는 Python 서버, 영상 분석, 음성 AI 연동, API 계약만 담당한다.
이미 구현된 핵심 기능은 다음과 같다.
FastAPI 서버
/health 상태 확인 API
/ws/vision WebSocket JPEG 프레임 수신
YOLO 기반 객체 탐지 및 추적
신호등, 버스, 키오스크, 표지판/화면 OCR 분석기 구조
VisionSession 기반 프레임 단위 분석
구조화된 analysis_events 생성
NarrationPolicy와 NarrationScheduler를 통한 한국어 안내 문장 생성
서버 응답에 narrations 텍스트 배열 포함
MP4 테스트 클라이언트 scripts/stream_video_client.py

현재 서버는 "음성 파일"을 만들지 않는다. 지금 구현된 것은 음성으로 읽을 수 있는 안내 문장 텍스트를 반환하는 단계다.

다음을 서버에 추가해야 한다.
사용자 음성 입력 수신 API
STT, LLM, TTS 흐름
Grok API 연동
Android 앱과 서버의 세션 연결
영상 분석 결과와 사용자 질문을 합쳐 답변하는 대화 로직
서버가 앱으로 음성 안내 결과를 돌려주는 API
인증/세션 토큰
실제 XIAO ESP32S3 Sense 또는 Android 앱과의 E2E 테스트
XIAO 펌웨어와 Android 앱 자체 구현은 이 레포의 주 작업 범위가 아니다. 다만 서버 API 계약과 테스트용 클라이언트는 이 레포에 추가해도 된다.

이번 작업의 목표는 "영상 분석 서버"를 "음성으로 질문하고 답을 받을 수 있는 서버"로 확장하는 것이다.
우선순위는 다음과 같다.
Android 앱이 사용할 서버 API 계약을 정리한다.
현재 /ws/vision에서 생성되는 최신 장면 분석결과를 세션별로 보관한다.
사용자 음성 또는 음성에서 변환된 텍스트를 받을 endpoint를 추가한다.
최신 장면 분석 결과와 사용자 질문을 Grok API에 전달한다.
Grok 응답을 앱에 반환한다.
가능하면 TTS 결과까지 반환하되, MVP에서는 텍스트 반환 후 Android TTS로 읽게 해도 된다.

권장 서버 구조
기존 /ws/vision은 유지한다. 여기에 음성/대화용 경로를 추가한다.
권장 API 예시는 다음과 같다.
POST /api/session
→ session_id 발급

WebSocket /ws/vision
→ 카메라 JPEG 프레임 수신
→ YOLO 분석
→ 최신 scene_state 저장
→ analysis_events, narrations 반환

POST /api/chat
→ Android 앱이 사용자 질문 텍스트를 전송
→ 서버가 최신 scene_state + question을 Grok API에 전달
→ answer_text 반환

선택:
WebSocket /ws/audio
→ Android 앱이 음성 chunk 전송
→ 서버 STT 후 /api/chat과 같은 흐름 수행

MVP에서는 /api/chat부터 구현하는 것을 권장한다. Android 앱에서 Buds 마이크 입력을 STT로 변환한 뒤 텍스트를 서버에 보내는 방식이 가장 빠르게 검증 가능하다.

사용자는 Grok API 키를 가지고 있다. 서버에서는 환경변수로 API 키를 읽도록 구현한다.

API 키를 코드에 하드코딩하지 말 것.

Grok에는 원본 영상을 계속 보내지 말고, 우선 현재 서버가 만든 구조화된 분석 결과를 전달한다.

예시 입력:

{
  "scene_state": {
    "recent_events": [
      {
        "object_type": "pedestrian_signal",
        "event_type": "object_state_changed",
        "current_state": "GREEN",
        "confidence": 0.86
      }
    ],
    "latest_narrations": [
      "보행자 신호가 초록색으로 바뀌었습니다."
    ]
  },
  "user_question": "지금 건너도 돼?"
}

응답은 짧고 안전하게 만든다. 특히 신호등/횡단 관련 답변은 직접적인 이동 명령을 피하고, 시스템이 본 근거와 불확실성을 함께 말해야 한다.

좋은 응답 예:
보행자 신호가 초록색으로 감지되었습니다. 다만 주변 차량이나 실제 도로 상황은 직접 확인이 필요합니다.

나쁜 응답 예:
지금 건너세요.

구현 시 주의사항

기존 YOLO 분석 파이프라인을 크게 갈아엎지 말 것.

/ws/vision의 기존 응답 형식을 깨지 말 것.

분석기 내부에서 Grok API, STT, TTS를 직접 호출하지 말 것.

영상 분석과 대화 AI를 분리할 것.

최신 장면 상태를 저장하는 별도 모듈을 둘 것.

API 키, 음성 파일, 원본 프레임을 로그에 남기지 말 것.

보행, 횡단, 차량 접근 같은 안전 관련 응답은 단정하지 말 것.

테스트 가능한 작은 단위로 구현할 것.

MVP 완료 기준

최소 완료 기준은 다음과 같다.

서버 실행 가능

/ws/vision 기존 테스트 통과

/api/chat 추가

/api/chat이 session_id와 user_question을 받음

서버가 해당 세션의 최신 분석 결과를 가져옴

Grok API client가 환경변수 기반으로 호출 가능

Grok 호출 실패 시 서버가 죽지 않고 명확한 에러 반환

테스트에서는 실제 Grok API를 호출하지 않고 fake client로 검증