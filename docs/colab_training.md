# Colab GPU에서 AIHub sidewalk 데이터로 YOLO 학습하기

로컬 서버(CPU 2코어)에서는 학습이 사실상 불가능하므로, 학습은 Colab GPU에서 하고
결과 가중치(best.pt)만 서버로 가져온다.

## 0. 사전 준비 (로컬에서 1회)

원본 데이터(이미지 + XML, 약 580MB)를 zip으로 묶어 Google Drive에 올린다.

```bash
cd ~/Voice_Agent_week_project/datasets/raw/aihub/sidewalk
zip -r sidewalk_train.zip train
```

`sidewalk_train.zip`을 Google Drive의 적당한 위치(예: `MyDrive/voice_agent/`)에 업로드한다.

## 1. Colab 노트북 설정

1. https://colab.research.google.com → 새 노트북
2. 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 선택
3. 아래 셀들을 순서대로 실행

## 2. 환경 준비 셀

```python
# GPU 확인 (Tesla T4가 보여야 함)
!nvidia-smi

# 레포 클론 + 의존성
!git clone https://github.com/Seojieun05/Voice_Agent_week_project.git
%cd Voice_Agent_week_project
!pip install -q ultralytics
```

## 3. 데이터 준비 셀

```python
# Drive 마운트 후 데이터 압축 해제
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p datasets/raw/aihub/sidewalk
!unzip -q /content/drive/MyDrive/voice_agent/sidewalk_train.zip -d datasets/raw/aihub/sidewalk
!ls datasets/raw/aihub/sidewalk/train | head
```

## 4. CVAT XML → YOLO 변환 셀

```python
!python scripts/convert_cvat_xml.py \
  --xml datasets/raw/aihub/sidewalk/train/*.xml \
  --images-root datasets/raw/aihub/sidewalk/train \
  --output-root datasets/processed/aihub_sidewalk \
  --class-mapping configs/aihub_sidewalk_classes.json \
  --split train --val-fraction 0.1 --link-mode copy
```

출력 JSON에서 `unknown_labels`가 비어 있고 train/val 분할(762/81)이 맞는지 확인한다.
`datasets/processed/aihub_sidewalk/previews/train/`의 preview 이미지로 박스가
물체에 잘 붙어 있는지 몇 장 눈으로 확인한다.

## 5. data.yaml 생성 셀

`configs/aihub_sidewalk_data.yaml`의 `path:`는 로컬 서버 절대경로라서 Colab에서는
새로 만드는 것이 안전하다:

```python
import yaml, pathlib
cfg = yaml.safe_load(open("configs/aihub_sidewalk_data.yaml"))
cfg["path"] = str(pathlib.Path("datasets/processed/aihub_sidewalk").resolve())
yaml.safe_dump(cfg, open("data_colab.yaml", "w"), sort_keys=False, allow_unicode=True)
print(open("data_colab.yaml").read())
```

## 6. 학습 셀

```python
# T4 기준 약 1~2시간. project를 Drive로 지정해 런타임이 끊겨도 체크포인트가 남게 한다.
!yolo detect train \
  data=data_colab.yaml \
  model=yolo26s.pt \
  epochs=100 imgsz=640 batch=16 device=0 \
  optimizer=SGD lr0=0.01 patience=30 \
  project=/content/drive/MyDrive/voice_agent/train name=aihub_sidewalk
```

- `yolo26s.pt`는 자동 다운로드된다.
- 중간에 끊기면 같은 명령에 `resume=True`를 붙여 이어서 학습한다.
- 목표 감각: car/pole/tree_trunk/bollard/person 같은 다수 클래스 mAP50 0.5+,
  전체 mAP50 0.3+ 정도면 이 데이터 규모(843장)에서 준수한 결과다.
  wheelchair/stroller/parking_meter처럼 표본이 10개 미만인 클래스는 0에 가깝게 나온다.

## 7. 검증 + 결과 확인 셀

```python
!yolo detect val \
  model=/content/drive/MyDrive/voice_agent/train/aihub_sidewalk/weights/best.pt \
  data=data_colab.yaml imgsz=640

# 샘플 이미지 추론 확인
!yolo predict \
  model=/content/drive/MyDrive/voice_agent/train/aihub_sidewalk/weights/best.pt \
  source=datasets/processed/aihub_sidewalk/images/val \
  conf=0.4 max_det=20 save=True
```

`runs/detect/predict/`의 결과 이미지에서 bollard, tree_trunk, movable_signage 등이
잘 잡히는지 확인한다.

## 8. 서버에 적용

1. Drive의 `train/aihub_sidewalk/weights/best.pt`를 다운로드해서 서버의
   `~/Voice_Agent_week_project/aihub_sidewalk_s.pt`로 복사한다 (`*.pt`는 gitignore 대상).
2. `.env`에 추가:

```
VISION_SERVER_MODEL=aihub_sidewalk_s.pt
```

3. 서버 재시작:

```bash
./run_server.sh
```

서버 코드는 클래스 이름 기반으로 동작하므로 수정 없이 새 클래스가 흡수된다:
`visible_objects`에 bollard/tree_trunk/pole 등이 등장하고, "앞으로 가도 돼?" 판단의
장애물 목록과 중요도 정렬(표지판·키오스크 그룹)에 자동 반영된다.

## 주의사항

- 학습 결과와 원본 데이터는 git에 커밋하지 않는다 (`datasets/`, `runs/`, `*.pt` 모두 ignore됨).
- 이 레포의 CPU 실험에서 확인한 함정 두 가지:
  - ultralytics는 data.yaml의 상대 `path`를 yaml 위치가 아니라 전역 `datasets_dir`
    기준으로 해석한다 → 절대경로 사용 (5번 셀 방식).
  - `optimizer=auto`는 fine-tune용 극소 lr(AdamW)을 골라 24-클래스 헤드가 거의
    학습되지 않았다 → `optimizer=SGD lr0=0.01` 명시.
