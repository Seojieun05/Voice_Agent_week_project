# Codex 다음 작업: 공개 테스트 영상 기반 평가 파이프라인 구축

아래 저장소의 현재 구현을 먼저 읽고 작업해줘.

- 저장소: `Voice_Agent_week_project`
- 현재 구조: YOLO26s → ByteTrack/stable ID → ObjectRouter → 객체별 Analyzer → SceneEventManager → NarrationPolicy
- 기존 신호등·버스·키오스크·텍스트·Generic VLM 분석기 구조를 유지할 것

이번 작업의 목표는 새로운 기능을 추가하는 것이 아니라, 제공된 공개 테스트 영상 팩에서 현재 모델의 동작을 재현 가능하게 측정하고 실패 조건을 기록하는 것이다.

## 입력 자료

테스트 팩을 프로젝트의 다음 경로에 풀었다고 가정한다.

```text
samples/public_test_pack/
├── sources.json
├── LICENSES.md
├── annotations/
├── videos/original/
└── videos/mp4/
```

영상이 `videos/mp4/`에 있으면 MP4를 우선 사용하고, 없으면 `videos/original/`의 OGV/WebM을 사용한다.
영상 파일은 Git에 커밋하지 않는다.

## 1. 먼저 현재 코드와 실제 입력 가능 클래스를 확인

다음을 코드 기준으로 확인하고 문서에 남겨라.

- 기본 YOLO26s COCO 모델이 실제로 출력할 수 있는 클래스
- `kiosk`, `sign`, `display`, `screen`, `unknown_panel` 라우트가 기본 모델에서 실제로 진입 가능한지
- 신호등 `signal_type`이 현재 `UNKNOWN`인지
- OCR과 Generic VLM이 활성화되는 조건
- 영상별로 어떤 Analyzer가 실제 호출됐는지

README의 주장만 복사하지 말고 파이프라인과 클래스 매핑 코드를 직접 대조하라.

## 2. 데이터셋 manifest와 annotation schema 구현

다음을 추가하라.

```text
datasets/public_baseline/
├── manifest.json
├── annotation.schema.json
├── annotations/
└── README.md
```

`manifest.json`은 `samples/public_test_pack/sources.json`에서 필요한 정보를 읽되 영상 경로는 상대 경로로 관리한다.

annotation은 최소 다음 필드를 지원해야 한다.

- video ID, FPS, frame count, width, height
- ground-truth object ID
- object type
- visible frame ranges
- 신호 상태 구간: RED/GREEN/YELLOW/OFF/UNKNOWN
- 명확한 상태 전환 frame과 ambiguous 여부
- 버스 motion 구간: APPROACHING/STOPPED/RECEDING/UNKNOWN
- 버스 route number 정답 또는 `null`
- 키오스크·화면 단계: ORDER_TYPE_SELECTION/PAYMENT/CONFIRMATION/UNKNOWN
- 금지해야 하는 발화 문장 또는 패턴
- 사람 검수 상태: `needs_manual_review`, `reviewed`

정답이 없는 값을 임의로 채우지 말고 `null` 또는 `needs_manual_review`로 남겨라.

## 3. annotation 보조 도구 구현

다음 CLI를 구현하라.

```bash
python scripts/prepare_annotations.py \
  --manifest datasets/public_baseline/manifest.json \
  --output-dir datasets/public_baseline/review
```

기능:

- 영상 metadata 자동 추출
- 일정 간격 contact sheet 생성
- frame 번호가 표시된 이미지 생성
- 영상별 annotation 초안 생성
- 기존 annotation이 있으면 덮어쓰지 않음
- 사람이 상태 경계와 객체 종류를 직접 검수할 수 있도록 README 생성

Codex가 영상 내용을 보고 정답 frame을 추측해 확정하지 말 것.

## 4. 현재 파이프라인 일괄 실행 도구 구현

다음 CLI를 구현하라.

```bash
python scripts/run_public_baseline.py \
  --manifest datasets/public_baseline/manifest.json \
  --output-dir outputs/public_baseline \
  --device cpu
```

기능:

- 각 영상마다 기존 `run_video_pipeline`을 호출
- 원본 출력 JSONL과 annotated MP4 보존
- 영상별 실행 설정을 기록
- 모델, imgsz, confidence, OCR 설정, VLM 설정, git commit SHA 기록
- 한 영상 실패가 전체 실행을 중단하지 않도록 오류를 영상별로 기록
- 실행 결과를 `run_summary.json`과 `run_summary.csv`에 저장

기본 설정을 영상마다 임의로 튜닝하지 말고 동일 baseline 설정으로 실행하라.

카테고리별 권장 클래스 필터:

- traffic-light 영상: COCO class 9
- bus 영상: COCO class 5
- kiosk-like/ticket-machine 영상: 기본 detector 전체 클래스 또는 별도 명시 설정

단, kiosk-like 영상에서 COCO detector가 `kiosk`를 출력하지 못하는 것은 숨기지 말고 baseline 한계로 기록하라.

## 5. 평가 스크립트 구현

다음 CLI를 구현하라.

```bash
python scripts/evaluate_public_baseline.py \
  --manifest datasets/public_baseline/manifest.json \
  --predictions outputs/public_baseline \
  --output-dir outputs/public_baseline/evaluation
```

사람 검수가 완료된 annotation에 대해서만 정량 평가하고, 미검수 영상은 qualitative 결과만 출력한다.

### 공통 지표

- 전체 처리 frame 수
- effective FPS
- realtime factor
- 평균 YOLO inference ms
- detection frame ratio
- stable ID fragmentation 수
- 발생 이벤트 수
- 중복 이벤트 수
- 생성된 발화 목록
- uncertain/UNKNOWN 비율

### 신호등 지표

- 상태별 frame accuracy
- RED/GREEN/YELLOW confusion matrix
- transition precision/recall
- 실제 transition과 모델 event의 frame·ms 지연
- false GREEN 확정 수
- duplicate transition 수
- subtype이 확인되지 않았는데 `보행자 신호`라고 발화한 수

### 버스 지표

- bus detection frame ratio
- track fragmentation
- APPROACHING precision/recall
- 접근 이벤트 중복 수
- route-number exact match
- 잘못 확정한 route number 수
- route number 확정 지연

### 키오스크·화면 지표

- 실제로 라우팅된 Analyzer 종류
- screen-stage accuracy
- OCR exact match 또는 normalized edit distance
- screen-change precision/recall
- 회수 기계를 주문 키오스크로 오분류한 수
- 고장 화면에서 PAYMENT/CONFIRMATION을 잘못 확정한 수

정답이 없는 지표는 0으로 가장하지 말고 `null`과 이유를 출력하라.

## 6. 영상별 기대되는 보수적 동작

- `signal_yellow_flicker_vertical`: YELLOW 또는 UNKNOWN은 허용하지만 확신한 RED/GREEN 전환 오탐은 실패로 기록
- `bus_waiting_multiple_arrivals`: 서로 다른 시점의 버스가 등장하며 stable ID reset과 중복 발화를 확인
- `bus_london_pulls_in`: 접근 이벤트는 최대 한 번, 읽을 수 없는 노선 번호는 확정하지 않음
- `kiosk_like_reverse_vending_machine`: 주문 키오스크 단계로 단정하지 않음
- `ticket_machine_defective_screen`: PAYMENT/CONFIRMATION 문장을 만들어내지 않음

이 기대 동작은 frame ground truth가 아니라 안전·환각 방지 constraint다.

## 7. 회귀 테스트 추가

실제 대형 영상은 단위 테스트에서 다운로드하거나 실행하지 않는다.

다음을 synthetic fixture와 fake model/OCR로 테스트하라.

- manifest/schema validation
- 검수되지 않은 annotation은 정량 평가에서 제외
- 정답 없는 metric은 `null`
- false GREEN 집계
- duplicate transition 집계
- route OCR exact-match 집계
- 잘못된 kiosk stage 집계
- 영상 하나 실패해도 batch가 계속되는지
- 결과 CSV와 JSON 필드 일치
- 실제 영상 파일을 Git에 추가하지 않는지

## 8. 지금 하지 말아야 하는 일

- TTS 구현
- 프런트엔드 또는 모바일 앱 구현
- 새로운 클라우드 API 추가
- Generic VLM을 모든 frame에 호출
- 라우터와 Analyzer 구조 전면 재작성
- 테스트 영상마다 HSV/YOLO threshold 개별 튜닝
- 정답이 없는 frame을 모델 출력으로 자동 라벨링해 ground truth로 사용
- 몇 개 영상 결과로 안전성이나 일반화를 주장
- 영상 원본을 Git 저장소에 커밋

## 9. 완료 조건

- `pytest -q` 전체 통과
- `ruff check .` 통과
- 7개 영상에 대한 batch 실행 가능
- 각 영상의 Analyzer routing과 한계를 확인할 수 있음
- annotation 검수 여부에 따라 정량/정성 평가가 구분됨
- `summary.json`, `summary.csv`, 영상별 report가 생성됨
- 기존 `detect_video.py`와 JSONL 출력 하위 호환 유지

## 10. 완료 후 보고 형식

1. 변경 파일 목록
2. dataset/annotation schema 설명
3. 추가한 CLI 사용법
4. 실제 실행한 명령
5. 영상별 라우팅 결과
6. 정량 평가 가능한 영상과 불가능한 영상 구분
7. false positive, false negative, UNKNOWN 사례
8. 현재 detector가 진입하지 못한 Router 경로
9. 기존 테스트와 새 테스트 결과
10. 다음 단계인 커스텀 detector 학습에 필요한 클래스와 데이터 목록

현재 코드를 먼저 읽고 최소 변경으로 구현하라. 평가 결과가 좋지 않아도 숨기거나 threshold를 즉흥적으로 조정하지 말고 그대로 기록하라.
