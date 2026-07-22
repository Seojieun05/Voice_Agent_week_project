#!/usr/bin/env python3
"""Download public test videos, validate them, optionally transcode, and build a ZIP.

No third-party Python package is required. ffmpeg/ffprobe are optional but strongly
recommended for standardized MP4 output and media validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCES_PATH = ROOT / "sources.json"
ORIGINAL_DIR = ROOT / "videos" / "original"
MP4_DIR = ROOT / "videos" / "mp4"
OUTPUT_ZIP = ROOT / "visual_assistant_test_videos.zip"


def load_manifest() -> dict[str, Any]:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {
        "User-Agent": "Voice-Agent-Vision-Test-Pack/1.0 (educational research)",
        "Accept": "*/*",
    }
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"    {downloaded / total:6.1%}", end="\r", flush=True)
            print(" " * 24, end="\r")
            temporary.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries} attempts: {url}: {error}") from error
            sleep_seconds = 2 ** (attempt - 1)
            print(f"    retry {attempt}/{retries} after {sleep_seconds}s: {error}")
            time.sleep(sleep_seconds)


def validate_download(path: Path, source: dict[str, Any], *, strict_size: bool) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"empty or missing file: {path}")
    actual_size = path.stat().st_size
    expected_size = source.get("expected_size_bytes")
    if strict_size and isinstance(expected_size, int):
        tolerance = max(4096, int(expected_size * 0.02))
        if abs(actual_size - expected_size) > tolerance:
            raise RuntimeError(
                f"unexpected size for {path.name}: got {actual_size}, expected about {expected_size}"
            )
    actual_sha1 = sha1(path)
    expected_sha1 = source.get("expected_sha1")
    if expected_sha1 and actual_sha1.lower() != str(expected_sha1).lower():
        raise RuntimeError(
            f"SHA-1 mismatch for {path.name}: got {actual_sha1}, expected {expected_sha1}"
        )
    return {"size_bytes": actual_size, "sha1": actual_sha1}


def run_ffprobe(path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def transcode_to_mp4(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vf", "scale='min(1280,iw)':-2:force_original_aspect_ratio=decrease,fps=30",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return True


def build_zip(*, include_originals: bool, include_mp4: bool) -> None:
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    static_paths = [
        ROOT / "README_KO.md",
        ROOT / "LICENSES.md",
        ROOT / "sources.json",
        ROOT / "codex_next_task.md",
        ROOT / "annotations" / "annotation_template.json",
    ]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in static_paths:
            archive.write(path, path.relative_to(ROOT))
        if include_originals:
            for path in sorted(ORIGINAL_DIR.glob("*")):
                if path.is_file() and not path.name.endswith(".part"):
                    archive.write(path, path.relative_to(ROOT))
        if include_mp4:
            for path in sorted(MP4_DIR.glob("*.mp4")):
                archive.write(path, path.relative_to(ROOT))
    print(f"Built: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List exact sources without downloading")
    parser.add_argument("--skip-transcode", action="store_true", help="Keep original OGV/WebM only")
    parser.add_argument("--originals-only", action="store_true", help="Download/validate only; do not ZIP")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument("--strict-size", action="store_true", help="Fail on source size drift over 2%")
    args = parser.parse_args()

    manifest = load_manifest()
    sources = manifest["sources"]
    if args.list:
        for item in sources:
            print(f"{item['id']}: {item['direct_url']} ({item['license']})")
        return 0

    results: list[dict[str, Any]] = []
    for index, item in enumerate(sources, start=1):
        original = ORIGINAL_DIR / item["download_filename"]
        print(f"[{index}/{len(sources)}] {item['id']}")
        if args.force or not original.exists():
            download(item["direct_url"], original)
        validation = validate_download(original, item, strict_size=args.strict_size)
        probe = run_ffprobe(original)
        result = {"id": item["id"], "original": original.name, **validation, "ffprobe": probe}

        if not args.skip_transcode:
            mp4 = MP4_DIR / item["standardized_filename"]
            if args.force or not mp4.exists():
                converted = transcode_to_mp4(original, mp4)
                if not converted:
                    print("    ffmpeg unavailable: standardized MP4 skipped")
            if mp4.exists():
                result["standardized_mp4"] = mp4.name
                result["standardized_sha1"] = sha1(mp4)
        results.append(result)

    (ROOT / "download_results.json").write_text(
        json.dumps({"sources": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.originals_only:
        build_zip(include_originals=True, include_mp4=not args.skip_transcode)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
