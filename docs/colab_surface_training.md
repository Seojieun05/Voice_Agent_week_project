# Colab GPU에서 노면(surface) 세그멘테이션 모델 학습하기

인도/차도/점자블록 등 **노면**을 구분하는 세그멘테이션 모델을 별도로 학습한다.
기존 객체 검출 모델(`aihub_sidewalk_bbox2_s.pt`)은 그대로 두고 두 모델을 병렬로 쓴다.
검출 모델 학습은 [colab_training.md](colab_training.md) 참고.

> **왜 한 모델로 합치지 않는가**
> AIHub의 Bbox 셋과 Surface 셋은 서로 다른 이미지다. 두 라벨을 한 데이터셋으로
> 합치면 Bbox 이미지에는 노면 폴리곤이, Surface 이미지에는 객체 박스가 없어서
> 학습 시 "여기엔 없다"는 잘못된 신호로 들어간다(partial annotation). 양쪽 성능이
> 같이 떨어지므로 데이터셋과 모델을 분리한다.

## 0. 사전 준비

surface zip 2개를 Google Drive에 올린다 (예: `MyDrive/voice_agent/surface_1.zip`,
`surface_2.zip`).

## 1. 환경 준비 셀

```python
!nvidia-smi   # Tesla T4 확인

!git clone https://github.com/Seojieun05/Voice_Agent_week_project.git
%cd Voice_Agent_week_project
!pip install -q ultralytics
```

## 2. 데이터 준비 셀 — zip 2개를 한 트리로 합친다

각 zip을 **서로 다른 하위 폴더**로 풀어야 한다. 같은 폴더로 풀면 두 아카이브에
동일한 scene 폴더 이름이 있을 때 조용히 덮어써진다.

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p datasets/raw/aihub/surface/part1 datasets/raw/aihub/surface/part2
!unzip -q /content/drive/MyDrive/voice_agent/surface_1.zip -d datasets/raw/aihub/surface/part1
!unzip -q /content/drive/MyDrive/voice_agent/surface_2.zip -d datasets/raw/aihub/surface/part2
!find datasets/raw/aihub/surface -maxdepth 3 -type d | head -20
```

## 3. 구조 점검 셀 — 변환 전에 반드시 실행

XML에 폴리곤이 들어 있는지, 라벨 이름이 무엇인지, MASK PNG의 색이 무엇인지 확인한다.

```python
!python scripts/convert_surface_xml.py \
  --roots datasets/raw/aihub/surface/part1 datasets/raw/aihub/surface/part2 \
  --inspect
```

출력에서 확인할 것:

- `XML child elements per image`에 **`polygon`이 있으면** 4-A로 간다 (권장).
  `polygon: 0`이거나 `box`만 있으면 4-B(MASK PNG)로 간다.
- `polygon labels` 목록이 [configs/aihub_surface_classes.json](../configs/aihub_surface_classes.json)의
  `classes`와 이름이 다르면, 그 파일의 `classes`를 실제 이름으로 고치거나
  `label_map`에 별칭을 넣는다. 예: `{"label_map": {"보도": "sidewalk"}}`.
  매핑되지 않은 라벨은 조용히 버려지지 않고 경고로 보고된다.
- `mask colors`는 4-B에서 팔레트를 만들 때 쓴다.

## 4-A. 변환 (XML 폴리곤 — 권장)

```python
!python scripts/convert_surface_xml.py \
  --roots datasets/raw/aihub/surface/part1 datasets/raw/aihub/surface/part2 \
  --output-root datasets/processed/aihub_surface \
  --class-mapping configs/aihub_surface_classes.json \
  --split train --val-fraction 0.1 --link-mode copy
```

두 아카이브가 하나의 데이터셋으로 합쳐지고, 출력 파일명이 `<scene>__<frame>` 으로
접두되므로 아카이브 간 파일명이 겹쳐도 충돌하지 않는다.

**val 분할은 프레임이 아니라 scene 단위**다. 한 scene의 연속 프레임은 거의 같은
그림이라, 프레임 단위로 나누면 val에 train과 똑같은 장면이 섞여 mAP가 실제보다
크게 부풀려진다. `split_scenes`로 어떤 scene이 val로 갔는지 확인할 수 있다.

## 4-B. 변환 (MASK PNG — XML에 폴리곤이 없을 때만)

3번 출력의 `mask colors`를 보고 팔레트 JSON을 만든다. 색은 `[R, G, B]`다.

```python
import json
palette = {
    "sidewalk": [255, 0, 0],          # ← inspect 출력의 실제 색으로 교체
    "braille_guide_blocks": [0, 255, 0],
    "roadway": [0, 0, 255],
}
json.dump({"palette": palette}, open("surface_palette.json", "w"), indent=2)
```

```python
!python scripts/convert_surface_xml.py \
  --roots datasets/raw/aihub/surface/part1 datasets/raw/aihub/surface/part2 \
  --output-root datasets/processed/aihub_surface \
  --class-mapping configs/aihub_surface_classes.json \
  --source mask --mask-palette surface_palette.json \
  --split train --val-fraction 0.1 --link-mode copy
```

## 5. 변환 결과 확인 셀

`previews/`에 폴리곤을 반투명으로 얹은 이미지가 저장된다. 인도가 인도 위에,
차도가 차도 위에 정확히 올라갔는지 눈으로 확인한다. 여기서 어긋나면 학습은 의미가 없다.

```python
from IPython.display import Image, display
import glob
for path in sorted(glob.glob("datasets/processed/aihub_surface/previews/*/*.jpg"))[:4]:
    print(path); display(Image(path, width=640))
```

`conversion_surface.json`에서 `unknown_labels`가 비어 있고 `missing_image_count`가
0인지도 확인한다.

## 6. data.yaml 생성 셀

```python
import yaml, pathlib
cfg = yaml.safe_load(open("configs/aihub_surface_data.yaml"))
cfg["path"] = str(pathlib.Path("datasets/processed/aihub_surface").resolve())
yaml.safe_dump(cfg, open("surface_colab.yaml", "w"), sort_keys=False, allow_unicode=True)
print(open("surface_colab.yaml").read())
```

`names:`가 3번에서 확인한 실제 클래스 순서와 일치하는지 마지막으로 대조한다.

## 7. 학습 셀

```python
!yolo segment train \
  data=surface_colab.yaml \
  model=yolo26n-seg.pt \
  epochs=100 imgsz=640 batch=16 device=0 \
  optimizer=SGD lr0=0.01 patience=30 \
  project=/content/drive/MyDrive/voice_agent/train name=aihub_surface
```

- `s-seg`가 아니라 **`n-seg`**를 쓰는 이유는 서버가 CPU 2코어이기 때문이다(8번 참고).
  노면은 덩어리가 크고 클래스가 적어서 `n`으로도 충분하다.
- 목표 감각: sidewalk/roadway 같은 큰 면적 클래스는 mask mAP50 0.7+, 점자블록처럼
  가늘고 긴 클래스는 0.4~0.5면 준수하다.
- 끊기면 같은 명령에 `resume=True`.

## 8. 검증 + 서버 적용

```python
!yolo segment val \
  model=/content/drive/MyDrive/voice_agent/train/aihub_surface/weights/best.pt \
  data=surface_colab.yaml imgsz=640

!yolo predict \
  model=/content/drive/MyDrive/voice_agent/train/aihub_surface/weights/best.pt \
  source=datasets/processed/aihub_surface/images/val conf=0.4 save=True
```

`best.pt`를 서버의 `~/Voice_Agent_week_project/aihub_surface_n_seg.pt`로 복사한다
(`*.pt`는 gitignore 대상).

### CPU 비용 — 실측

서버(Intel Xeon 2코어)에서 `yolo26n-seg` 추론 시간:

| imgsz | 프레임당 |
|---|---|
| 320 | 51 ms |
| 480 | 77 ms |
| 640 | 135 ms |

객체 검출 모델이 이미 매 프레임 코어를 쓰고 있으므로, **노면 모델은 매 프레임 돌리면
안 된다.** 노면은 초 단위로 바뀌는 정보가 아니므로 `imgsz=320`에 10프레임마다 1회
(30fps 기준 3Hz) 정도면 충분하고, 추가 부하는 실질 5% 수준이다.

파이프라인에 붙일 때 [src/vision_agent/pipeline.py](../src/vision_agent/pipeline.py)의
`PipelineConfig`에 `surface_model` / `surface_interval_frames` / `surface_image_size`를
추가하고, `.env`로 켜고 끌 수 있게 한다:

```
VISION_SERVER_SURFACE_MODEL=aihub_surface_n_seg.pt
VISION_SERVER_SURFACE_INTERVAL_FRAMES=10
VISION_SERVER_SURFACE_IMAGE_SIZE=320
```
