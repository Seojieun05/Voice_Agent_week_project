# Codex 작업 지침 — Voice Agent Week Project

이 문서는 Codex가 이 저장소의 목적과 현재 상태를 이해하고, 범위를 벗어나지 않으면서 다음 작업을 수행하도록 안내한다.

## 1. 프로젝트 목적

이 프로젝트는 **시각장애인을 위한 능동형 시각 보조 시스템**의 비전 파이프라인을 만드는 프로젝트다.

최종 목표는 안경형 카메라 또는 스마트폰 카메라 영상을 지속적으로 관찰하면서 다음과 같은 중요한 변화를 감지하고, 필요한 정보만 사용자에게 먼저 안내하는 것이다.

- 보행자 신호등의 상태와 상태 전환
- 접근하는 버스와 버스 번호
- 키오스크 화면의 단계 및 선택지
- 표지판, 출입구, 엘리베이터 등 주변 환경 정보

현재 단계에서는 전체 서비스가 아니라 **YOLO26 기반 객체 탐지·추적 MVP**만 구현한다.

현재 파이프라인의 범위는 다음과 같다.

```text
MP4 영상 입력
→ YOLO26 객체 탐지
→ ByteTrack 객체 추적
→ 안정적인 등장·사라짐 이벤트 생성
→ 바운딩박스 영상 및 JSONL 로그 저장
```

## 2. 현재 구현 상태

현재 저장소에는 다음 기능이 구현되어 있다.

- `yolo26n.pt` 사전학습 모델 로드
- MP4 영상 프레임 단위 추론
- Ultralytics `model.track(..., persist=True)`를 통한 ByteTrack 추적
- COCO 클래스 필터링
- 탐지 결과가 표시된 MP4 저장
- 프레임별 탐지 결과 및 이벤트 JSONL 저장
- 객체가 여러 프레임에서 안정적으로 관찰되었을 때 `appeared` 이벤트 생성
- 일정 프레임 동안 관찰되지 않았을 때 `disappeared` 이벤트 생성
- 평균 추론 시간 및 전체 처리 FPS 출력

주요 파일:

```text
scripts/detect_video.py              CLI 진입점
scripts/check_environment.py         Python/PyTorch/CUDA 환경 확인
src/vision_agent/pipeline.py         영상 처리 및 YOLO 실행 파이프라인
src/vision_agent/events.py           등장·사라짐 이벤트 안정화
src/vision_agent/types.py            Detection 및 SceneEvent 자료형
src/vision_agent/io.py               JSONL 출력
 tests/test_events.py                이벤트 엔진 단위 테스트
```

현재 프로젝트의 객체 분석 구조를 아래 방향으로 설계하고 구현해줘.

## 프로젝트 목적

이 프로젝트는 시각장애인이 착용하는 카메라 또는 스마트폰 카메라 영상을 지속적으로 분석하고, 사용자에게 필요한 시각 정보를 능동적으로 음성 안내하는 시스템이다.

현재 YOLO26 기반 객체 탐지·추적 파이프라인과 신호등 상태 분석 기능을 구현하고 있다.

이 프로젝트에서는 범용 비전 모델이 모든 객체를 자유롭게 설명하도록 만들지 않는다.

신호등, 버스, 키오스크, 표지판처럼 서비스의 핵심 대상은 객체 종류를 먼저 분류한 뒤, 각 객체에 맞는 전용 분석 로직을 실행해야 한다.

분류할 수 없는 객체에만 범용 비전 모델을 fallback으로 사용한다.

## 핵심 설계 원칙

전체 장면을 하나의 종류로 분류하지 말고, 화면 안의 객체를 각각 탐지하고 추적해야 한다.

예를 들어 한 화면에 다음 객체가 동시에 존재할 수 있다.

* 보행자 신호등
* 차량용 신호등
* 버스
* 키오스크
* 표지판
* 사람
* 자동차

각 객체는 `stable_id`를 가져야 하며, 프레임이 바뀌어도 같은 객체라면 같은 ID로 유지해야 한다.

객체가 탐지되면 다음과 같이 객체 종류에 따라 분석기를 라우팅한다.

```text
카메라 프레임
    ↓
YOLO26 객체 탐지
    ↓
객체 추적 및 stable ID 부여
    ↓
Object Router
    ├─ traffic_light → TrafficLightAnalyzer
    ├─ bus → BusAnalyzer
    ├─ kiosk → KioskAnalyzer
    ├─ sign/display/screen → TextObjectAnalyzer
    └─ unknown → GenericVisionAnalyzer
    ↓
구조화된 분석 결과
    ↓
Scene Event Manager
    ↓
Narration Policy
    ↓
TTS
```

## 객체 라우터 구현

객체 종류에 따라 적절한 분석기를 선택하는 구조를 구현해줘.

예시:

```python
def route_detection(detection):
    if detection.class_name == "traffic_light":
        return traffic_light_analyzer.analyze(detection)

    if detection.class_name == "bus":
        return bus_analyzer.analyze(detection)

    if detection.class_name == "kiosk":
        return kiosk_analyzer.analyze(detection)

    if detection.class_name in {"sign", "display", "screen"}:
        return text_object_analyzer.analyze(detection)

    return generic_vision_analyzer.analyze(detection)
```

단순한 `if` 문에 모든 기능을 넣지 말고, 분석기별 클래스를 분리하고 공통 인터페이스를 정의해줘.

예상 구조:

```text
src/vision_agent/
├── analyzers/
│   ├── base.py
│   ├── traffic_light.py
│   ├── bus.py
│   ├── kiosk.py
│   ├── text_object.py
│   └── generic.py
├── router.py
├── events.py
├── narration.py
├── pipeline.py
└── types.py
```

## 공통 분석 결과 형식

각 분석기는 자연어 문장을 바로 반환하지 말고, 구조화된 결과를 반환해야 한다.

공통 데이터 모델 예시:

```python
@dataclass
class AnalysisResult:
    object_type: str
    stable_id: str
    state: str | None
    confidence: float
    attributes: dict[str, object]
    is_uncertain: bool
```

객체별 추가 정보는 `attributes` 안에 저장한다.

분석 결과에는 반드시 다음 정보가 포함되어야 한다.

* 객체 종류
* stable ID
* 분석 상태
* 신뢰도
* 불확실 여부
* 객체별 추가 속성

## 신호등 분석

신호등은 자유로운 자연어 설명 모델에 맡기지 않는다.

신호등 분석기는 다음을 판단해야 한다.

* 보행자용인지 차량용인지
* 현재 상태가 RED, GREEN, YELLOW, UNKNOWN 중 무엇인지
* 이전 상태에서 변경되었는지
* 몇 프레임 연속 동일 상태가 확인됐는지
* 분석 근거가 충분한지

예시 결과:

```json
{
  "object_type": "pedestrian_signal",
  "stable_id": "stable-1",
  "state": "RED",
  "confidence": 0.94,
  "attributes": {
    "previous_state": "GREEN",
    "changed": true,
    "confirmed_frames": 3
  },
  "is_uncertain": false
}
```

신호등 상태는 연속 프레임으로 확인해야 한다.

```text
GREEN, GREEN, GREEN
→ GREEN 확정

RED, GREEN, RED
→ 상태 유지 또는 UNKNOWN

RED, RED, RED
→ RED 전환 확정
```

불확실한 경우 임의로 상태를 선택하지 말고 `UNKNOWN`을 반환한다.

## 버스 분석

버스 분석기는 다음을 담당한다.

* 동일 버스 추적
* 버스가 접근 중인지, 멀어지는지, 정차 중인지 판단
* 버스 번호 표시 영역 추출
* OCR 결과 안정화
* 여러 프레임에서 동일한 번호가 확인됐을 때만 확정

예시 결과:

```json
{
  "object_type": "bus",
  "stable_id": "stable-7",
  "state": "APPROACHING",
  "confidence": 0.91,
  "attributes": {
    "route_number": "3102",
    "route_confidence": 0.89,
    "ocr_confirmed_frames": 4
  },
  "is_uncertain": false
}
```

OCR 결과가 프레임마다 다르면 버스 번호를 확정하지 않는다.

```text
310?
3102
3102
3102
→ 3102 확정
```

## 키오스크 분석

키오스크 분석기는 다음을 담당한다.

* 키오스크 또는 주문 화면 탐지
* 화면 영역 추출
* OCR을 통한 버튼 및 문구 인식
* 현재 화면 단계 판단
* 이전 화면과 비교하여 화면 변경 감지

예시 결과:

```json
{
  "object_type": "kiosk",
  "stable_id": "stable-12",
  "state": "ORDER_TYPE_SELECTION",
  "confidence": 0.87,
  "attributes": {
    "visible_options": [
      "매장 식사",
      "포장"
    ],
    "screen_changed": true
  },
  "is_uncertain": false
}
```

키오스크 분석은 초기 구현에서 완성된 VLM 연동까지 만들지 말고, 인터페이스와 placeholder 구현부터 만들어도 된다.

## 표지판 및 화면 문자 분석

표지판, 전광판, 화면은 OCR 중심으로 처리한다.

다음 조건을 만족할 때만 문자를 확정한다.

* 글자 영역이 충분히 큼
* OCR 신뢰도가 기준 이상
* 여러 프레임에서 결과가 일치
* 이전에 안내한 내용과 다름

글자가 불분명한 경우 내용을 추측하지 않는다.

## Generic Vision fallback

알려진 객체 분석기에 해당하지 않는 객체에만 범용 비전 모델을 사용할 수 있도록 인터페이스를 만든다.

현재 단계에서는 실제 외부 VLM API를 연결하지 말고, 비활성화된 placeholder 또는 명시적인 `NotImplemented` 구현으로 둔다.

Generic Vision fallback은 다음 용도로만 사용한다.

* 분류되지 않은 새로운 기계나 시설
* 임시 공사 안내물
* 알 수 없는 조작 패널
* 서비스에 사전 정의되지 않은 객체

신호등, 버스 번호, 안전 관련 상태 판단에는 Generic Vision fallback을 사용하지 않는다.

## Scene Event Manager

분석 결과를 바로 음성으로 말하지 말고, 먼저 이벤트로 변환해야 한다.

지원할 이벤트 예시:

```text
OBJECT_APPEARED
OBJECT_DISAPPEARED
OBJECT_STATE_CHANGED
TEXT_CONFIRMED
SCREEN_CHANGED
OBJECT_APPROACHING
```

예시:

```json
{
  "event_type": "OBJECT_STATE_CHANGED",
  "object_type": "pedestrian_signal",
  "stable_id": "stable-1",
  "previous_state": "GREEN",
  "current_state": "RED",
  "timestamp_s": 2.733
}
```

같은 이벤트가 반복 발생하지 않도록 중복 억제 기능을 구현한다.

## Narration Policy

이벤트가 발생했다고 무조건 말하지 않는다.

다음 기준으로 발화 여부를 결정하는 구조를 만들어줘.

* 사용자 안전과 직접 관련된가
* 이전에 이미 안내한 정보인가
* 새로운 정보 또는 상태 변화인가
* 분석 신뢰도가 충분한가
* 최근 발화와 중복되는가
* 다른 더 중요한 이벤트가 동시에 발생했는가

우선순위 예시:

```text
1. 보행자 신호 상태 변화
2. 접근하는 차량 또는 버스
3. 버스 번호 확정
4. 키오스크 화면 변경
5. 표지판 문자 확정
6. 일반 객체 등장
```

## 문장 생성 방식

신호등, 버스 번호처럼 구조화된 객체는 LLM이 자유롭게 문장을 생성하지 않도록 한다.

정형 템플릿을 사용한다.

예시:

```python
if event.object_type == "pedestrian_signal":
    if event.current_state == "GREEN":
        message = "보행자 신호가 초록색으로 바뀌었습니다."
    elif event.current_state == "RED":
        message = "보행자 신호가 빨간색으로 바뀌었습니다."
```

버스:

```python
message = f"{route_number}번 버스가 들어오고 있습니다."
```

키오스크:

```python
message = "매장 식사와 포장 중 하나를 선택하는 화면입니다."
```

불확실한 정보를 포함한 문장을 생성하지 않는다.

## 지금 구현해야 하는 범위

이번 작업에서는 다음을 구현한다.

1. 객체별 Analyzer 공통 인터페이스
2. Object Router
3. 기존 신호등 상태 분석 코드를 `TrafficLightAnalyzer`로 분리
4. `BusAnalyzer`, `KioskAnalyzer`, `TextObjectAnalyzer`, `GenericVisionAnalyzer` 기본 구조
5. 구조화된 `AnalysisResult` 타입
6. 분석 결과에서 이벤트를 생성하는 기본 Event Manager
7. 정형 템플릿 기반 Narration Policy
8. 단위 테스트
9. 기존 신호등 영상 파이프라인과 호환 유지

버스 OCR, 키오스크 OCR, 범용 VLM 호출은 인터페이스와 placeholder까지만 구현해도 된다.

## 하지 말아야 하는 일

다음 작업은 하지 마.

* 전체 장면을 신호등 장면, 버스 장면처럼 하나로만 분류하지 말 것
* 모든 객체를 범용 VLM에 보내지 말 것
* 신호등 상태를 자연어 생성 모델에 맡기지 말 것
* 불확실한 글자, 버스 번호, 장소명을 추측하지 말 것
* 분석기 내부에서 바로 TTS를 실행하지 말 것
* 탐지 결과가 나온 프레임마다 반복 발화하지 말 것
* 신호등 색상 판정을 위해 무조건 가장 많은 색상을 선택하지 말 것
* UNKNOWN 상태를 실패로 취급하지 말 것
* 기존 YOLO 탐지·stable ID·신호 전환 기능을 깨뜨리지 말 것
* 현재 단계에서 프런트엔드나 모바일 앱을 만들지 말 것
* 새로운 외부 유료 API를 임의로 추가하지 말 것
* MiniCPM을 다시 핵심 분석 모델로 추가하지 말 것

## 테스트 요구사항

다음 테스트를 추가해줘.

### Router 테스트

* traffic light가 `TrafficLightAnalyzer`로 전달되는지
* bus가 `BusAnalyzer`로 전달되는지
* kiosk가 `KioskAnalyzer`로 전달되는지
* 알 수 없는 클래스가 `GenericVisionAnalyzer`로 전달되는지

### 신호등 테스트

* GREEN 3프레임 후 GREEN 확정
* GREEN 이후 RED 3프레임 후 전환 이벤트 1회
* RED가 계속 유지될 때 추가 이벤트가 발생하지 않음
* 불확실한 프레임이 섞이면 잘못된 전환 이벤트가 발생하지 않음

### Narration 테스트

* GREEN → RED 이벤트에서 지정 문장 1개 생성
* 동일 이벤트가 반복돼도 중복 문장이 생성되지 않음
* UNKNOWN 상태에서는 안전 관련 확정 문장이 생성되지 않음

### 기존 기능 회귀 테스트

* 기존 영상에서 259/259프레임 탐지 결과 유지
* stable ID가 stable-1로 유지
* GREEN → RED 이벤트가 정확히 한 번만 발생
* 기존 JSONL 및 결과 영상 생성 기능 유지

## 완료 후 보고

작업이 끝나면 다음 형식으로 보고해줘.

1. 변경한 파일 목록
2. 각 파일의 역할
3. 구현한 분석기와 라우팅 구조
4. Event Manager 동작 방식
5. Narration Policy 동작 방식
6. 실행한 테스트와 결과
7. 기존 기능 회귀 여부
8. 아직 placeholder로 남아 있는 기능
9. 다음으로 구현해야 할 기능

범위를 임의로 넓히지 말고, 기존 프로젝트 구조를 먼저 읽은 뒤 최소 변경으로 구현해줘.
