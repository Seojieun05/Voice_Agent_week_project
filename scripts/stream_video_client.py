#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MP4 프레임을 /ws/vision 서버에 JPEG로 전송하는 최소 검증 클라이언트입니다."
    )
    parser.add_argument("--source", required=True, help="전송할 로컬 MP4 경로")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/vision")
    parser.add_argument("--session-id", default="sample-session")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="전송 FPS. 생략하면 영상 FPS를 사용하고 읽을 수 없으면 15를 사용",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 영상 끝까지 전송")
    return parser


async def stream_video(args: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "WebSocket 클라이언트 의존성이 없습니다. `pip install -e '.[server]'`를 실행하세요."
        ) from exc

    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {source}")
    if args.fps is not None and args.fps <= 0.0:
        raise ValueError("--fps는 0보다 커야 합니다.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality는 1부터 100 사이여야 합니다.")
    if args.max_frames < 0:
        raise ValueError("--max-frames는 0 이상이어야 합니다.")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"영상의 첫 프레임을 읽을 수 없습니다: {source}")
        height, width = first_frame.shape[:2]
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        send_fps = args.fps or (source_fps if source_fps > 0.0 else 15.0)
        frame_interval_s = 1.0 / send_fps

        async with websockets.connect(args.url, max_size=16 * 1024 * 1024) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "session_id": args.session_id,
                        "source_width": width,
                        "source_height": height,
                        "source_fps": send_fps,
                    },
                    ensure_ascii=False,
                )
            )

            sequence_id = 0
            frame = first_frame
            while frame is not None and (args.max_frames == 0 or sequence_id < args.max_frames):
                loop_started_at_s = time.perf_counter()
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not encoded_ok:
                    raise RuntimeError(f"frame {sequence_id} JPEG 인코딩에 실패했습니다.")
                captured_at_ms = time.time_ns() // 1_000_000
                await websocket.send(
                    json.dumps(
                        {
                            "type": "frame",
                            "sequence_id": sequence_id,
                            "captured_at_ms": captured_at_ms,
                        }
                    )
                )
                await websocket.send(encoded.tobytes())
                response = json.loads(await websocket.recv())
                print(json.dumps(response, ensure_ascii=False))

                sequence_id += 1
                ok, next_frame = capture.read()
                frame = next_frame if ok else None
                remaining_s = frame_interval_s - (time.perf_counter() - loop_started_at_s)
                if remaining_s > 0.0:
                    await asyncio.sleep(remaining_s)
    finally:
        capture.release()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(stream_video(args))
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"오류: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
