# 초기 COCO 클래스

YOLO26s 기본 가중치는 COCO 80개 클래스를 사용합니다.
이 프로젝트에서 먼저 확인할 클래스 ID는 다음과 같습니다.

- `5`: bus
- `9`: traffic light
- `11`: stop sign
- `62`: tv
- `0`: person
- `2`: car

주의: 기본 `traffic light` 클래스는 보행자/차량용이나 표시 색을 구분하지 않습니다.
현재 색 상태는 선택된 대표 박스의 별도 HSV 분석기가 판정하며, 보행자용/차량용 구분은
아직 `UNKNOWN` placeholder입니다.

객체 라우터는 기본 COCO의 `bus`를 버스 접근·번호 분석기로, `stop sign`과 `tv`를
문자 분석기로 보냅니다. `kiosk`, `sign`, `display`, `screen`은 COCO 기본 클래스가
아니므로 해당 이름을 출력하도록 학습된 커스텀 탐지 모델이 필요합니다. 나머지 클래스는
Generic Vision 모델과 `--vlm-classes` allowlist에 모두 명시된 경우에만 fallback 설명을
생성합니다.
