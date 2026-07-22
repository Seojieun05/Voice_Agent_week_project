#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vision_agent.public_baseline.runner import BaselineSettings, run_public_baseline


def _parse_class_names(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("--vlm-classes에는 클래스 이름을 하나 이상 입력하세요.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공개 테스트 팩의 모든 영상을 동일 설정으로 실행합니다."
    )
    parser.add_argument("--manifest", required=True, help="public baseline manifest JSON")
    parser.add_argument("--output-dir", required=True, help="배치 결과 저장 폴더")
    parser.add_argument("--device", default=None, help="추론 장치 (예: cpu, 0, cuda:0)")
    parser.add_argument("--model", default="yolo26s.pt", help="공통 YOLO 체크포인트")
    parser.add_argument("--imgsz", type=int, default=640, help="공통 모델 입력 크기")
    parser.add_argument("--conf", type=float, default=0.10, help="공통 confidence 임계값")
    parser.add_argument("--no-track", action="store_true", help="모든 영상에서 추적 비활성화")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="공통 tracker 설정")
    parser.add_argument(
        "--ocr-backend",
        choices=("rapidocr", "none"),
        default="rapidocr",
        help="공통 OCR 백엔드",
    )
    parser.add_argument("--ocr-language", default="default", help="공통 RapidOCR 언어")
    parser.add_argument("--ocr-model-path", default=None, help="공통 오프라인 OCR 모델")
    parser.add_argument(
        "--allow-ocr-download",
        action="store_true",
        help="모든 영상에서 OCR 모델 다운로드 허용",
    )
    parser.add_argument("--vlm-model", default=None, help="공통 Generic VLM 모델")
    parser.add_argument("--vlm-device", default=None, help="공통 Generic VLM 장치")
    parser.add_argument(
        "--vlm-classes",
        type=_parse_class_names,
        default=("unknown", "unknown_object", "unknown_panel"),
        help="Generic VLM 허용 클래스 이름",
    )
    parser.add_argument(
        "--allow-vlm-download",
        action="store_true",
        help="모든 영상에서 Generic VLM 다운로드 허용",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = BaselineSettings(
        model=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        track=not args.no_track,
        tracker=args.tracker,
        ocr_backend=args.ocr_backend,
        ocr_language=args.ocr_language,
        ocr_model_path=args.ocr_model_path,
        allow_ocr_download=args.allow_ocr_download,
        generic_vlm_model=args.vlm_model,
        generic_vlm_device=args.vlm_device,
        allow_vlm_download=args.allow_vlm_download,
        generic_vlm_classes=args.vlm_classes,
    )
    try:
        result = run_public_baseline(
            Path(args.manifest),
            Path(args.output_dir),
            settings=settings,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "total_videos": result.summary["total_videos"],
                "succeeded_videos": result.summary["succeeded_videos"],
                "failed_videos": result.summary["failed_videos"],
                "run_summary_json": str(result.json_path),
                "run_summary_csv": str(result.csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result.failed_videos else 0


if __name__ == "__main__":
    raise SystemExit(main())
