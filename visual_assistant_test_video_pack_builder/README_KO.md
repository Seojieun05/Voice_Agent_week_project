# 시각 보조 프로젝트 공개 테스트 영상 팩 빌더

## 중요한 설명

이 ZIP에는 저작권·용량·현재 실행 환경의 외부 바이너리 다운로드 제한 때문에 **영상 원본을 직접 동봉하지 않았습니다.**
대신 라이선스가 명확하고 URL이 고정된 Wikimedia Commons 원본 7개를 정확히 지정하고,
한 번의 명령으로 다운로드·무결성 검사·MP4 변환·최종 ZIP 생성을 수행하는 빌더를 포함합니다.

## 선정 구성

- 신호등 3개
  - 보행자 신호의 비교적 긴 연속 영상
  - 세로형·카운트다운 보행 신호
  - 황색 점멸 세로 영상
- 버스 2개
  - 한 영상에서 두 버스가 서로 다른 시점에 등장
  - 버스가 정류장으로 진입하는 연속 영상
- 키오스크·기계 화면 2개
  - 사람이 조작하는 회수 기계: `kiosk` 오분류 방지와 Generic fallback 시험
  - 고장 난 발권기 화면: OCR·UNKNOWN·환각 방지 시험

모든 영상은 편집된 광고나 하이라이트가 아니라 단일 연속 촬영에 가깝습니다. 다만 공개 자료의 한계로
모두가 안경형 카메라 POV는 아닙니다. 최종 일반화 검증에는 사용자가 직접 촬영한 걷는 시점 영상이 별도로 필요합니다.

## 실행

Linux/SSH 서버에서:

```bash
unzip visual_assistant_test_video_pack_builder.zip
cd visual_assistant_test_video_pack_builder
python3 download_and_build.py
```

`ffmpeg`가 설치돼 있으면 모든 영상을 H.264 MP4, 최대 1280px, 30FPS, 무음으로 표준화합니다.
완료 후 다음 파일이 생성됩니다.

```text
visual_assistant_test_videos.zip
videos/original/*
videos/mp4/*.mp4
download_results.json
```

ffmpeg가 없다면:

```bash
python3 download_and_build.py --skip-transcode
```

출처만 확인하려면:

```bash
python3 download_and_build.py --list
```

## 프로젝트에서 사용

최종 생성된 `visual_assistant_test_videos.zip`을 프로젝트 외부에서 풀거나,
`.gitignore`된 `samples/public_test_pack/` 아래에 복사하세요. 큰 영상 파일은 Git에 커밋하지 않는 것을 권장합니다.

```bash
mkdir -p samples/public_test_pack
unzip visual_assistant_test_videos.zip -d samples/public_test_pack
```

## 테스트 해석

이 영상들은 기능을 깨뜨리는 조건을 찾기 위한 baseline입니다.
몇 개 영상에서 성공했다고 안전성·일반화 성능을 주장하면 안 됩니다.
특히 다음은 실패가 아니라 바람직한 보수적 결과일 수 있습니다.

- 증거가 부족한 신호 상태를 `UNKNOWN`으로 반환
- 읽을 수 없는 버스 번호를 확정하지 않음
- 회수 기계를 주문 키오스크로 분류하지 않음
- 고장 난 화면에서 결제·확인 단계를 만들어내지 않음

`annotations/annotation_template.json`을 영상별로 복사한 뒤 사람이 직접 프레임 정답을 작성해야
정확도·전환 지연·OCR exact match를 계산할 수 있습니다.
