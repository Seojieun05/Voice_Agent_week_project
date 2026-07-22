#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vision_agent.public_baseline.evaluation import EvaluationError, evaluate_public_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공개 baseline JSONL을 사람 검수 annotation과 보수적으로 평가합니다."
    )
    parser.add_argument("--manifest", required=True, help="dataset manifest JSON 경로")
    parser.add_argument("--predictions", required=True, help="run_public_baseline 출력 폴더")
    parser.add_argument("--output-dir", required=True, help="summary와 영상별 report 저장 폴더")
    parser.add_argument(
        "--transition-tolerance-frames",
        type=int,
        default=30,
        help="GT와 예측 transition을 대응시킬 최대 frame 거리 (기본: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = evaluate_public_baseline(
            Path(args.manifest),
            Path(args.predictions),
            Path(args.output_dir),
            transition_tolerance_frames=args.transition_tolerance_frames,
        )
    except (EvaluationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
