# Voice Agent Week Project

안경형 카메라를 위한 능동형 시각 보조 서비스의 비전 MVP입니다. 현재 단계에서는
**YOLO26s + ByteTrack**으로 영상 속 객체를 탐지·추적하고, 한 신호등의 바운딩박스를
안정적으로 선택한 뒤 crop 기반의 실험적 HSV 분류기로 `RED`, `GREEN`, `YELLOW`,
`UNKNOWN`을 판정합니다. 연속된 결과가 충분히 쌓였을 때만 상태 전환 이벤트를
기록합니다.

## 현재 구현 범위

- 기본값 `yolo26s.pt`, 입력 크기 `640`, confidence `0.10`
- 로컬 MP4 영상의 YOLO26 탐지 및 ByteTrack 추적
- 짧은 track ID 단절을 IoU로 재연결하는 stable object ID
- 한 신호등을 구성하는 중복 바운딩박스를 묶고 대표 target 하나를 선택
- 선택한 신호등 crop의 실험적 HSV `RED` / `GREEN` / `YELLOW` / `UNKNOWN` 분류
- 연속 상태를 이용한 `signal_changed` 이벤트
- 안정적인 등장·사라짐 이벤트
- stable ID와 객체 종류에 따른 Analyzer 라우팅
- 버스 bbox 변화 기반 접근·정차·후퇴 판정과 번호 OCR 연속 확인
- 키오스크 OCR 기반 단계·버튼 인식과 화면 변경 확인
- 표지판·전광판·화면 OCR의 크기·신뢰도·연속 프레임 확인
- 알 수 없는 객체 crop만 처리하는 선택적 로컬 Generic VLM fallback
- 공통 `AnalysisResult`와 분석 이벤트 JSON 직렬화
- 이벤트 우선순위·신뢰도·중복 억제를 적용하는 정형 Narration Policy
- 선택적 신호등 crop 저장
- 바운딩박스가 표시된 결과 영상과 프레임별 JSONL 저장
- 영상 PTS 기반 단조 비감소 타임스탬프와 처리 성능 요약
- CUDA 사용 가능 여부에 따른 GPU 0/CPU 자동 선택

> HSV 상태 분류기는 `visiontest2.mp4`와 `visiontest3.mp4`에서 검증한 실험 기능입니다.
> 조명, 카메라, 신호등 형태가 달라지면 오판할 수 있으며, 색 근거가 약하거나 서로 충돌하면
> `UNKNOWN`을 반환합니다. 이 결과로 안전한 횡단 여부를 판단하거나 횡단을 지시해서는
> 안 됩니다.

## 객체별 분석 구조

```text
YOLO26 탐지 → ByteTrack/stable ID → ObjectRouter
    ├─ traffic light       → TrafficLightAnalyzer
    ├─ bus                 → BusAnalyzer
    ├─ kiosk               → KioskAnalyzer
    ├─ sign/display/screen → TextObjectAnalyzer
    └─ 그 외               → GenericVisionAnalyzer
→ AnalysisResult → SceneEventManager → NarrationPolicy
```

`TrafficLightAnalyzer`는 stable ID마다 연속 관측 이력을 따로 관리합니다. 파이프라인에서
이미 계산한 HSV 결과를 재사용하므로 같은 crop을 중복 분류하지 않으며, 세 프레임이
확정되기 전에는 `is_uncertain=true`로 반환합니다. 탐지 프레임이 끊기면 전환 후보의
연속 횟수도 초기화됩니다. 기존 `StableObjectEventEngine` 이벤트는 원래 시각 그대로 먼저
변환되고, `SceneEventManager`가 Analyzer의 확정 상태 전환을 보완한 뒤 같은 전환은 한
번만 남깁니다.

`BusAnalyzer`는 stable ID별 bbox 면적과 중심 이동을 파이프라인 기본 9프레임의 중앙값
추세로 비교해 `APPROACHING`, `STOPPED`, `RECEDING`, `UNKNOWN`을 안정화합니다. 작은 bbox
흔들림과 최대 2프레임의 누락·저신뢰 관측은 허용하지만 긴 단절에서는 이력을 초기화합니다.
번호 OCR은 접근 또는 정차가 확정된 버스에만 기본 7프레임 간격으로 실행합니다. 한 프레임에
숫자가 여러 개면 최고 신뢰 후보가 차점 후보보다 충분히 우세할 때만 투표하며, 실제로 새로
실행한 OCR에서 같은 번호가 3회 확인돼야 확정합니다. 현재 관측이 끊기면 과거 번호는 진단
이력으로만 남고 접근 발화에 다시 사용하지 않습니다. `KioskAnalyzer`는 OCR 문구와 버튼을
이용해 주문 방식 선택·결제·확인 단계를 판정하고, 화면 지문이 연속 확인됐을 때만 변경
이벤트를 만듭니다. `TextObjectAnalyzer`는 충분히 큰 crop에서 신뢰도 기준을 통과한 문자가
충분히 큰 OCR bbox를 가지며 기본 3프레임 일치할 때만 확정합니다. 비슷해 보여도 숫자가
다른 문자열은 하나의 후보로 합치지 않습니다.

`GenericVisionAnalyzer`는 위 전용 분석기에 매핑되지 않고 `--vlm-classes` allowlist에도
명시된 객체 crop에만 사용합니다. 로컬 Transformers `image-text-to-text` 모델을 지정한
경우에만 활성화되며, 추론 간격을 제한하고 기본적으로 동일한 설명을 2회 확인한 뒤
`DESCRIPTION_CONFIRMED` 이벤트를 생성합니다. 신호등, 버스 번호, OCR 텍스트 같은 전용
판단을 대신하지 않습니다. 모든 분석기는 근거가 부족하면 내용을 추측하지 않고 `UNKNOWN`
또는 `is_uncertain=true`를 반환합니다.

버스 모션·번호 OCR은 detector confidence가 기본 `0.30` 이상일 때 이력을 쌓고, 키오스크·
문자·Generic 분석은 기본 `0.50` 이상을 요구합니다. 확정 결과 confidence는 detector와
OCR/VLM 근거 중 낮은 값을 사용합니다. 전체 YOLO 임계값 `0.10`은 작은 객체를 놓치지 않기
위한 값이며, 도메인별 기준에 못 미치는 저신뢰 객체는 분석 발화를 만들지 않습니다.

키오스크의 `visible_options`와 `recognized_buttons`는 매장/포장, 결제 수단, 확인/취소처럼
지원하는 단계·버튼 키워드만 담습니다. 그 밖의 짧은 bbox OCR 문구는 버튼으로 단정하지 않고
`button_candidates`에 text/confidence/bbox와 함께 보존합니다. 최초 화면도 안정화된 뒤 한 번
안내하며, 단계가 `UNKNOWN`이어도 화면 자체의 OCR 지문이 확정되면 화면 이벤트는 생성됩니다.

## 1. 설치

```bash
cd Voice_Agent_week_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

GPU가 있는 서버라면 PyTorch가 CUDA를 인식하는지 확인합니다.

```bash
python scripts/check_environment.py
```

`cuda_available`이 `true`면 GPU 추론이 가능합니다.

### 선택적 OCR 설치

버스 번호, 키오스크, 표지판·화면 OCR에는 RapidOCR와 ONNX Runtime이 필요합니다.

```bash
pip install -e '.[ocr]'
```

기본 `default` 인식 모델은 패키지에 포함된 가중치를 사용하므로 숫자·라틴 문자는 오프라인에서도
동작하며, 네트워크에서 OCR 모델을 임의로 내려받지 않습니다. 한국어 PP-OCRv5 인식 모델 파일을
준비했다면 다음처럼 지정합니다.

```bash
python scripts/detect_video.py \
  --source samples/bus.mp4 \
  --classes 5 \
  --ocr-model-path /models/korean-rec.onnx
```

RapidOCR의 모델 다운로드를 허용하려면 `--allow-ocr-download`를 명시합니다. OCR 패키지나
모델을 사용할 수 없어도 영상 처리는 계속되며 해당 분석만 불확실 결과로 기록됩니다.

### `visiontest.mp4` 버스 검증

COCO 버스 클래스 ID `5`를 지정해야 버스 분석기가 실행됩니다.

```bash
python scripts/detect_video.py \
  --source samples/visiontest.mp4 \
  --classes 5 \
  --device cpu \
  --output-dir outputs/visiontest_bus_complete
```

저장소의 영상 전체 301프레임을 실행하면 버스가 282프레임에서 탐지되고, 후반부 버스의 접근
이벤트가 약 5.80초에 한 번 생성됩니다. 기본 OCR은 `532`를 연속 확인해 약 6.50초에 번호를
확정합니다. 결과 영상과 프레임별 근거는 각각 `visiontest_annotated.mp4`,
`visiontest_detections.jsonl`에 저장됩니다.

### 선택적 Generic VLM 설치

```bash
pip install -e '.[vlm]'
python scripts/detect_video.py \
  --source samples/scene.mp4 \
  --vlm-model /models/local-image-text-model \
  --vlm-device cpu \
  --vlm-classes "unknown panel,vending machine"
```

`--vlm-model`을 생략하면 Generic VLM은 비활성화됩니다. 기본 allowlist는 `unknown`,
`unknown_object`, `unknown_panel`이며 다른 클래스는 `--vlm-classes`에 명시해야 합니다.
모델 ID를 사용해 원격 가중치를 받으려면 `--allow-vlm-download`를 함께 지정해야 합니다.
외부 유료 API는 연결하지 않으며, 모델과 OCR 엔진은 해당 객체가 실제로 등장할 때까지
지연 로드됩니다. chat/instruction 형식의 `image-text-to-text` pipeline을 지원하는 모델을
사용해야 합니다.

## 2. 신호등 영상 실행

```bash
python scripts/detect_video.py \
  --source samples/visiontest2.mp4 \
  --classes 9 \
  --save-crops
```

`--classes 9`는 COCO의 `traffic light`만 탐지합니다. 모델·입력 크기·confidence를
생략하면 검증된 기본값인 `yolo26s.pt`, `640`, `0.10`을 사용합니다. 첫 실행에서는
모델 가중치가 자동 다운로드될 수 있습니다.

`--device`를 생략하면 CUDA 사용 가능 여부에 따라 GPU 0 또는 CPU를 자동 선택합니다.
CPU를 명시하려면 `--device cpu`, GPU 0을 명시하려면 `--device 0`을 사용합니다.
CUDA가 없는 환경에서 GPU를 강제하면 CPU로 조용히 전환하지 않고 해결 방법이 포함된
오류를 출력합니다.

### 탐지 모델 선택 근거

`visiontest2.mp4`의 259프레임을 CPU에서 비교한 결과입니다.

| 설정 | 신호등이 탐지된 프레임 | 최대 연속 누락 |
| --- | ---: | ---: |
| YOLO26n, 1280, conf 0.10 | 252 / 259 (97.3%) | 5프레임 |
| YOLO26s, 640, conf 0.10 | 259 / 259 (100%) | 0프레임 |

따라서 기본 모델을 YOLO26s로 조정했습니다. 이 수치는 제공된 한 영상에 대한 결과이며
다른 촬영 조건에서의 일반 성능을 뜻하지 않습니다.

### 대표 신호등 target

COCO 모델은 한 신호등에서 상단 표시부, 숫자 카운트다운, 전체 구조를 서로 다른
`traffic light` 박스로 내놓을 수 있습니다. 파이프라인은 겹치거나 세로로 인접한
구성요소를 같은 신호등 그룹으로 묶고, 이전 프레임의 원본 track ID를 우선 유지해
대표 박스 하나만 상태 판정·이벤트·주석에 사용합니다. 공간적으로 떨어진 실제 다른
신호등은 별도로 유지합니다. 원본 탐지는 분석을 위해 JSONL에 모두 보존됩니다.

### 연속 상태 전환 판정

- YOLO 탐지는 작은 신호등 보존을 위해 `0.10`부터 기록하지만, 색 상태 이력에는 detector
  confidence가 기본 `0.20` 이상인 crop만 사용합니다.
- 알려진 같은 상태가 기본 3프레임 연속이면 기준 상태로 확정합니다.
- 반대 상태가 다시 3프레임 연속일 때만 `signal_changed`를 한 번 발생시킵니다.
- `UNKNOWN`이나 일시 누락은 후보 연속 횟수를 끊지만 마지막 확정 상태는 보존합니다.
- 연속 프레임 수는 `--min-signal-state-frames`로 변경할 수 있습니다.

### 실제 전환 시점 검증

| 영상 | 원본에서 명확한 상태 변화 | 모델 이벤트 | 차이 |
| --- | --- | --- | ---: |
| `visiontest2.mp4` | frame 80, 2.667초 `GREEN → RED` | frame 82, 2.733초 | +0.067초 |
| `visiontest3.mp4` | frame 28, 1.120초 `RED → GREEN` | frame 29, 1.160초 | +0.040초 |

`visiontest3`의 frame 25~27은 빨강 소등과 초록 점등이 겹치는 전환 구간입니다. frame 28을
최초의 명확한 GREEN으로 판독했습니다. 모델은 frame 27~29를 GREEN으로 세 번 확인한 뒤
이벤트를 기록하므로, 명확한 상태 경계보다 40ms 늦고 최초 초록 점등(frame 25)보다
160ms 늦습니다.

이 영상에는 정면 보행자 신호 외에 불빛이 보이지 않는 옆면 차량 신호도 있습니다.
전역 `signal_state_counts`는 공간적으로 분리된 모든 신호를 합산하므로 차량 신호의
`UNKNOWN`도 포함합니다. 정면 보행 신호만 보면 `RED 26 / UNKNOWN 1 / GREEN 246`입니다.
정면 신호 박스는 273/282프레임에서 검출됐지만 사람 가림으로 frame 221~229에 끊기며,
가림 뒤 stable ID도 `stable-2`에서 `stable-4`로 바뀝니다. 따라서 `visiontest3`은 상태
전환은 맞지만 영상 전체의 바운딩박스·stable ID 연속성까지 만족하는 회귀 영상은 아닙니다.

## 3. 결과

`--save-crops`를 사용한 실행은 다음 파일을 생성합니다.

```text
outputs/
├── visiontest2_annotated.mp4
├── visiontest2_detections.jsonl
└── visiontest2_crops/
    └── frame_000012__traffic-light-stable-1__conf_0.900.jpg
```

crop 저장은 기본적으로 꺼져 있습니다. crop 좌표는 프레임 경계 안으로 제한되며,
선택된 대표 신호등만 저장합니다.

JSONL 한 줄은 한 프레임의 결과입니다. `track_id`는 YOLO/ByteTrack의 원본 ID이고,
`stable_object_key`는 이벤트 엔진이 재연결한 ID입니다.

```json
{
  "frame_index": 82,
  "timestamp_s": 2.7333333333333334,
  "inference_ms": 192.148,
  "detections": [
    {
      "class_id": 9,
      "class_name": "traffic light",
      "confidence": 0.3912428319454193,
      "track_id": 1,
      "xyxy": [347.1799, 251.8982, 413.8858, 389.1013],
      "stable_object_key": "traffic light:stable-1",
      "is_signal_target": true,
      "signal_state": "RED",
      "signal_state_confidence": 1.0,
      "signal_red_ratio": 0.141657,
      "signal_green_ratio": 0.0,
      "signal_yellow_ratio": 0.0,
      "analysis": {
        "object_type": "traffic_light",
        "stable_id": "stable-1",
        "state": "RED",
        "confidence": 1.0,
        "attributes": {
          "previous_state": "GREEN",
          "changed": true,
          "confirmed_frames": 3,
          "signal_type": "UNKNOWN"
        },
        "is_uncertain": false
      }
    }
  ],
  "events": [
    {
      "event_type": "signal_changed",
      "object_key": "traffic light:stable-1",
      "timestamp_s": 2.7333333333333334,
      "message": "신호등 표시가 초록색에서 빨간색으로 바뀌었습니다.",
      "previous_state": "GREEN",
      "current_state": "RED"
    }
  ],
  "analysis_results": [
    {
      "object_type": "traffic_light",
      "stable_id": "stable-1",
      "state": "RED",
      "confidence": 1.0,
      "attributes": {
        "previous_state": "GREEN",
        "changed": true,
        "confirmed_frames": 3,
        "signal_type": "UNKNOWN"
      },
      "is_uncertain": false
    }
  ],
  "analysis_events": [
    {
      "event_type": "OBJECT_STATE_CHANGED",
      "object_type": "traffic_light",
      "stable_id": "stable-1",
      "timestamp_s": 2.7333333333333334,
      "previous_state": "GREEN",
      "current_state": "RED",
      "confidence": 1.0,
      "attributes": {},
      "is_uncertain": false
    }
  ],
  "narrations": [
    "신호등 표시가 빨간색으로 바뀌었습니다."
  ]
}
```

기존 소비자를 위해 원래의 `detections`와 `events` 필드는 그대로 유지됩니다. 새
`analysis_results`, `analysis_events`, `narrations`는 하위 호환 방식으로 추가됩니다.
내레이션은 문자열 후보만 생성하며 실제 음성 합성은 수행하지 않습니다.

실행 종료 요약에는 `video_duration_s`, `source_fps`, `frames`, `elapsed_s`,
`effective_fps`, `realtime_factor`, `average_inference_ms`, `classes`, `ocr_backend`,
`ocr_language`, 신호등 집계, `bus_detection_frames`, `bus_motion_state_counts`,
`bus_approach_events`, `bus_route_numbers`, 분석 이벤트·내레이션 및 출력 경로가 포함됩니다.
`realtime_factor <= 1.0`이면 영상 길이와 같거나 더 빠른 처리입니다.

## 4. 주요 옵션

```text
--model                         기본 yolo26s.pt
--classes                       탐지할 COCO ID 목록, 예: 5,9
--conf                          confidence 임계값, 기본 0.10
--imgsz                         모델 입력 크기, 기본 640
--device                        생략 시 CUDA 0/CPU 자동 선택
--no-track                      추적 없이 프레임별 탐지만 수행
--min-seen-frames               등장 이벤트 확정 연속 프레임 수
--max-missed-frames             사라짐 이벤트 확정 누락 프레임 수
--reconnect-iou-threshold       track ID 재연결 최소 IoU
--max-reconnect-frames          새 ID 재연결 최대 누락 프레임 수
--bus-motion-window-frames      버스 bbox 추세 관측 수, 기본 9
--bus-minimum-detection-confidence 버스 분석 최소 confidence, 기본 0.30
--bus-minimum-area-change-ratio 접근·이탈 최소 면적 변화율, 기본 0.10
--bus-max-motion-frame-gap      버스 모션 이력 허용 누락, 기본 2
--bus-route-ocr-interval-frames 번호 OCR 호출 간격, 기본 7
--no-signal-state               HSV 신호 상태 분류 끄기
--min-signal-state-frames       상태 확정·전환 최소 연속 프레임 수
--signal-minimum-detection-confidence 색상 이력용 최소 탐지 confidence, 기본 0.20
--signal-minimum-color-ratio    crop 내 최소 색상 픽셀 비율, 기본 0.015
--signal-minimum-score-margin   우세 색상과 다른 색상 점수의 최소 차이
--signal-minimum-dominance-ratio 우세 색상의 최소 비율
--save-crops                    선택된 신호등 crop 저장
--ocr-backend                   rapidocr 또는 none
--ocr-language                  OCR 인식 언어, 기본 내장 default
--ocr-model-path                오프라인 RapidOCR 인식 모델 파일
--allow-ocr-download            RapidOCR 모델 다운로드 허용
--vlm-model                     로컬 VLM 경로 또는 Transformers 모델 ID
--vlm-device                    Generic VLM 추론 장치
--vlm-classes                   Generic VLM 호출을 허용할 클래스 이름 목록
--allow-vlm-download            원격 VLM 가중치 다운로드 허용
```

## 5. 테스트

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

단위 테스트는 모델 다운로드나 네트워크 연결을 사용하지 않습니다.

## 6. 현재 제한사항

- COCO `traffic light` 클래스는 보행자용과 차량용 신호등을 구분하지 않습니다.
- HSV 임계값은 `visiontest2.mp4`와 `visiontest3.mp4`에서 검증한 실험값이며 다양한
  조명·카메라·신호 형태에 일반화되었다고 볼 수 없습니다.
- `RED`, `GREEN`, `YELLOW`, 상태 전환 결과는 관측 정보일 뿐 횡단 안전 판단이나 행동
  지시가 아닙니다.
- stable ID 재연결은 같은 클래스의 바운딩박스 IoU와 짧은 누락 구간을 사용합니다.
- `visiontest3`처럼 `max_missed_frames`보다 긴 가림에서는 같은 물리 객체라도 새 stable ID가
  부여될 수 있습니다.
- COCO 클래스만으로 보행자용/차량용 신호를 확정할 수 없어 `signal_type`은 현재
  `UNKNOWN`입니다.
- 버스 접근 판정은 한 카메라의 bbox 크기·중심 변화 휴리스틱이므로 카메라 자체의 이동,
  가림, 급격한 시점 변화에 영향을 받습니다.
- 기본 COCO 모델에는 `kiosk`, `sign`, `display`, `screen` 클래스가 없습니다. 해당 이름을
  출력하는 커스텀 탐지 모델이 있어야 전용 라우팅이 가능하며, COCO의 `bus`, `stop sign`,
  `tv`는 각각 버스 또는 문자 분석기로 라우팅됩니다.
- OCR 품질은 crop 해상도와 선택한 언어 모델에 의존합니다. 버스는 `visiontest.mp4` 한 편에서
  실제 검증했지만 다양한 노선 표시·야간·가림 조건에 일반화됐다고 볼 수 없습니다. 키오스크는
  실제 촬영 영상이 없어 합성 crop과 주입형 OCR 결과 중심으로 검증했습니다.
- 키오스크 버튼 후보는 OCR bbox가 있는 짧은 문구이며 버튼 외 제목이 포함될 수 있습니다.
  지원 키워드가 아닌 후보는 확정 버튼으로 발화하지 않습니다.
- Transformers 생성 파이프라인은 보정된 확률을 제공하지 않아 Generic VLM confidence는
  설정된 가정값입니다. 따라서 연속 결과 확인 후에도 안전 판단에는 사용할 수 없습니다.
- TTS와 실시간 카메라 입력은 이번 구현 범위에 포함하지 않습니다.

## 라이선스 주의

Ultralytics YOLO26과 `ultralytics` 패키지는 AGPL-3.0 또는 별도 Enterprise 라이선스를
사용합니다. 연구·학교 프로젝트를 넘어 비공개 상용 서비스로 확장할 때는 라이선스를
다시 검토해야 합니다.
