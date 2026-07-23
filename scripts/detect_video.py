#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vision_agent.pipeline import PipelineConfig, run_video_pipeline


def _parse_classes(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    try:
        return tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--classes에는 쉼표로 구분한 정수 ID를 입력하세요."
        ) from exc


def _parse_class_names(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("--vlm-classes에는 클래스 이름을 하나 이상 입력하세요.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO26으로 영상 객체를 탐지하고 결과 영상·JSONL 로그를 저장합니다."
    )
    parser.add_argument("--source", required=True, help="입력 영상 경로")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO 체크포인트")
    parser.add_argument("--output-dir", default="outputs", help="결과 저장 폴더")
    parser.add_argument(
        "--classes",
        default=None,
        help="COCO 클래스 ID. 예: 신호등만 탐지하려면 9, 버스와 신호등은 5,9",
    )
    parser.add_argument("--conf", type=float, default=0.10, help="confidence 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="모델 입력 크기")
    parser.add_argument(
        "--device",
        default=None,
        help="추론 장치. 생략하면 CUDA 0 또는 CPU를 자동 선택 (예: 0, cpu, cuda:0)",
    )
    parser.add_argument("--no-track", action="store_true", help="ByteTrack을 사용하지 않음")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker 설정")
    parser.add_argument("--min-seen-frames", type=int, default=3)
    parser.add_argument("--max-missed-frames", type=int, default=8)
    parser.add_argument(
        "--reconnect-iou-threshold",
        type=float,
        default=0.3,
        help="track ID 재연결을 위한 최소 IoU",
    )
    parser.add_argument(
        "--max-reconnect-frames",
        type=int,
        default=3,
        help="새 track ID를 기존 객체에 재연결할 수 있는 최대 누락 프레임",
    )
    parser.add_argument(
        "--bus-motion-window-frames",
        type=int,
        default=9,
        help="버스 접근·이탈 추세에 사용할 관측 프레임 수",
    )
    parser.add_argument(
        "--bus-minimum-detection-confidence",
        type=float,
        default=0.3,
        help="버스 모션·번호 OCR 이력에 사용할 최소 탐지 confidence",
    )
    parser.add_argument(
        "--bus-minimum-area-change-ratio",
        type=float,
        default=0.1,
        help="버스 접근·이탈 판정에 필요한 추세 구간 bbox 면적 변화율",
    )
    parser.add_argument(
        "--bus-max-motion-frame-gap",
        type=int,
        default=2,
        help="버스 모션 이력을 유지할 수 있는 최대 누락·저신뢰 프레임 수",
    )
    parser.add_argument(
        "--bus-route-ocr-interval-frames",
        type=int,
        default=7,
        help="접근·정차 버스 번호 OCR 호출 사이의 최소 프레임 수",
    )
    parser.add_argument(
        "--kiosk-ocr-interval-frames",
        type=int,
        default=1,
        help="키오스크 OCR 호출 사이의 최소 처리 프레임 수",
    )
    parser.add_argument(
        "--text-ocr-interval-frames",
        type=int,
        default=1,
        help="표지판·화면 OCR 호출 사이의 최소 처리 프레임 수",
    )
    parser.add_argument(
        "--no-signal-state",
        action="store_true",
        help="실험용 빨강/초록/노랑/알 수 없음 분류를 비활성화",
    )
    parser.add_argument(
        "--min-signal-state-frames",
        type=int,
        default=3,
        help="신호 상태 확정·전환에 필요한 연속 프레임",
    )
    parser.add_argument(
        "--signal-minimum-detection-confidence",
        type=float,
        default=0.2,
        help="신호 색상 이력에 사용할 최소 신호등 탐지 confidence",
    )
    parser.add_argument(
        "--signal-minimum-color-ratio",
        type=float,
        default=0.015,
        help="RED/GREEN/YELLOW 판정에 필요한 ROI 내 최소 색상 비율",
    )
    parser.add_argument(
        "--signal-minimum-score-margin",
        type=float,
        default=0.015,
        help="우세 색상과 반대 색상의 최소 비율 차이",
    )
    parser.add_argument(
        "--signal-minimum-dominance-ratio",
        type=float,
        default=2.0,
        help="우세 색상이 반대 색상보다 커야 하는 최소 배수",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="선택된 stable 신호등 crop을 디버그 이미지로 저장",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=("rapidocr", "none"),
        default="rapidocr",
        help="버스·키오스크·표지판 OCR 백엔드 (기본: rapidocr)",
    )
    parser.add_argument(
        "--ocr-language",
        choices=(
            "default",
            "ch",
            "ch_doc",
            "en",
            "arabic",
            "chinese_cht",
            "cyrillic",
            "devanagari",
            "japan",
            "korean",
            "ka",
            "latin",
            "ta",
            "te",
            "eslav",
            "th",
            "el",
        ),
        default="default",
        help="RapidOCR 인식 언어 (기본: 내장 default; 한국어 모델은 별도 경로 필요)",
    )
    parser.add_argument(
        "--ocr-model-path",
        default=None,
        help="오프라인 RapidOCR 인식 모델 파일 경로",
    )
    parser.add_argument(
        "--allow-ocr-download",
        action="store_true",
        help="RapidOCR이 선택한 언어 모델을 다운로드하도록 명시적으로 허용",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Generic Vision fallback용 로컬 Transformers 모델 경로 또는 모델 ID",
    )
    parser.add_argument(
        "--vlm-device",
        default=None,
        help="Generic VLM 추론 장치 (예: cpu, cuda:0)",
    )
    parser.add_argument(
        "--vlm-classes",
        type=_parse_class_names,
        default=("unknown", "unknown_object", "unknown_panel"),
        help="Generic VLM을 허용할 클래스 이름 목록",
    )
    parser.add_argument(
        "--allow-vlm-download",
        action="store_true",
        help="--vlm-model의 원격 가중치 다운로드를 명시적으로 허용",
    )
    parser.add_argument(
        "--narrate-presence-classes",
        type=_parse_class_names,
        default=(),
        help="등장·사라짐 발화를 명시적으로 허용할 객체 클래스 목록",
    )
    parser.add_argument(
        "--no-bus-approach-narration",
        action="store_true",
        help="버스 접근 이벤트는 기록하되 발화하지 않음",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        classes = _parse_classes(args.classes)
        config = PipelineConfig(
            source=args.source,
            model=args.model,
            output_dir=Path(args.output_dir),
            classes=classes,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
            track=not args.no_track,
            tracker=args.tracker,
            min_seen_frames=args.min_seen_frames,
            max_missed_frames=args.max_missed_frames,
            reconnect_iou_threshold=args.reconnect_iou_threshold,
            max_reconnect_frames=args.max_reconnect_frames,
            bus_motion_window_frames=args.bus_motion_window_frames,
            bus_minimum_detection_confidence=(args.bus_minimum_detection_confidence),
            bus_minimum_area_change_ratio=args.bus_minimum_area_change_ratio,
            bus_maximum_motion_frame_gap=args.bus_max_motion_frame_gap,
            bus_route_ocr_interval_frames=args.bus_route_ocr_interval_frames,
            kiosk_ocr_interval_frames=args.kiosk_ocr_interval_frames,
            text_ocr_interval_frames=args.text_ocr_interval_frames,
            classify_signal_states=not args.no_signal_state,
            min_signal_state_frames=args.min_signal_state_frames,
            signal_minimum_detection_confidence=(args.signal_minimum_detection_confidence),
            signal_minimum_color_ratio=args.signal_minimum_color_ratio,
            signal_minimum_score_margin=args.signal_minimum_score_margin,
            signal_minimum_dominance_ratio=args.signal_minimum_dominance_ratio,
            save_crops=args.save_crops,
            ocr_backend=args.ocr_backend,
            ocr_language=args.ocr_language,
            ocr_model_path=args.ocr_model_path,
            allow_ocr_download=args.allow_ocr_download,
            generic_vlm_model=args.vlm_model,
            generic_vlm_device=args.vlm_device,
            allow_vlm_download=args.allow_vlm_download,
            generic_vlm_classes=args.vlm_classes,
            narration_presence_classes=args.narrate_presence_classes,
            narrate_bus_approach=not args.no_bus_approach_narration,
        )
        summary = run_video_pipeline(config)
    except (FileNotFoundError, RuntimeError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
