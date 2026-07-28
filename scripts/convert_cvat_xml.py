"""Convert CVAT XML annotations (e.g. AIHub sidewalk) into YOLO format.

Reads one or more CVAT ``annotations`` XML files, maps source labels to a
unified class list, and writes ``class_id x_center y_center width height``
label files plus copied/symlinked images in the standard YOLO layout:

    output_root/images/<split>/<image>
    output_root/labels/<split>/<stem>.txt
    output_root/previews/<split>/<stem>.jpg   (first few, boxes drawn)
    output_root/conversion_<split>.json       (statistics)

Unknown labels are never silently mapped; they are excluded and reported
with the XML files they came from. Duplicate image filenames found under
different folders are a hard error rather than a silent pick.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ClassMapping:
    """Unified class list plus source-label aliases."""

    classes: tuple[str, ...]
    label_map: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "ClassMapping":
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = tuple(str(name) for name in payload.get("classes", ()))
        if not classes:
            raise ValueError(f"{path}: 'classes' must be a non-empty list")
        if len(set(classes)) != len(classes):
            raise ValueError(f"{path}: 'classes' contains duplicates")
        label_map = {
            str(key): str(value) for key, value in dict(payload.get("label_map", {})).items()
        }
        for alias, target in label_map.items():
            if target not in classes:
                raise ValueError(f"{path}: label_map '{alias}' points to unknown class '{target}'")
        return cls(classes=classes, label_map=label_map)

    def class_id(self, label: str) -> int | None:
        canonical = self.label_map.get(label, label)
        try:
            return self.classes.index(canonical)
        except ValueError:
            return None


@dataclass(frozen=True)
class CvatBox:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float


@dataclass(frozen=True)
class CvatImage:
    name: str
    width: int
    height: int
    boxes: tuple[CvatBox, ...]
    source_xml: Path


@dataclass
class ConversionStats:
    converted_images: int = 0
    converted_boxes: int = 0
    empty_images: int = 0
    missing_images: list[str] = field(default_factory=list)
    excluded_invalid_boxes: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)
    unknown_label_sources: dict[str, list[str]] = field(default_factory=dict)
    source_class_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "converted_images": self.converted_images,
            "converted_boxes": self.converted_boxes,
            "empty_images": self.empty_images,
            "missing_images": sorted(self.missing_images),
            "missing_image_count": len(self.missing_images),
            "excluded_invalid_boxes": self.excluded_invalid_boxes,
            "unknown_labels": dict(sorted(self.unknown_labels.items())),
            "unknown_label_sources": {
                label: sorted(paths) for label, paths in sorted(self.unknown_label_sources.items())
            },
            "source_class_counts": dict(sorted(self.source_class_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
        }


def parse_cvat_xml(xml_path: Path) -> list[CvatImage]:
    """Parse every ``.//image`` element of one CVAT XML file."""
    root = ET.parse(xml_path).getroot()
    images: list[CvatImage] = []
    for element in root.findall(".//image"):
        name = element.get("name")
        width = element.get("width")
        height = element.get("height")
        if not name or width is None or height is None:
            raise ValueError(f"{xml_path}: image element missing name/width/height")
        boxes = []
        for box in element.findall("box"):
            label = box.get("label")
            if label is None:
                raise ValueError(f"{xml_path}: box without label in image '{name}'")
            boxes.append(
                CvatBox(
                    label=label,
                    xtl=float(box.get("xtl", "nan")),
                    ytl=float(box.get("ytl", "nan")),
                    xbr=float(box.get("xbr", "nan")),
                    ybr=float(box.get("ybr", "nan")),
                )
            )
        images.append(
            CvatImage(
                name=Path(name).name,
                width=int(width),
                height=int(height),
                boxes=tuple(boxes),
                source_xml=xml_path,
            )
        )
    return images


def convert_box(
    box: CvatBox,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Clamp to the image and return normalized YOLO xywh, or None if invalid."""
    xtl = min(max(box.xtl, 0.0), float(image_width))
    ytl = min(max(box.ytl, 0.0), float(image_height))
    xbr = min(max(box.xbr, 0.0), float(image_width))
    ybr = min(max(box.ybr, 0.0), float(image_height))
    if xbr <= xtl or ybr <= ytl:
        return None
    x_center = (xtl + xbr) / 2.0 / image_width
    y_center = (ytl + ybr) / 2.0 / image_height
    width = (xbr - xtl) / image_width
    height = (ybr - ytl) / image_height
    return (x_center, y_center, width, height)


def index_images(images_root: Path) -> dict[str, Path]:
    """Recursively index image files by filename; duplicates are an error."""
    found: dict[str, list[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found[path.name].append(path)
    duplicates = {name: paths for name, paths in found.items() if len(paths) > 1}
    if duplicates:
        lines = [
            f"  {name}: " + ", ".join(str(path) for path in sorted(paths))
            for name, paths in sorted(duplicates.items())
        ]
        raise SystemExit(
            "duplicate image filenames under --images-root; refusing to pick silently:\n"
            + "\n".join(lines)
        )
    return {name: paths[0] for name, paths in found.items()}


def assign_split(stem: str, base_split: str, val_fraction: float) -> str:
    """Deterministically route a stem to '<base_split>' or 'val'."""
    if val_fraction <= 0.0:
        return base_split
    bucket = zlib.crc32(stem.encode("utf-8")) % 1000
    return "val" if bucket < int(val_fraction * 1000) else base_split


def _place_image(source: Path, target: Path, link_mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if link_mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def _write_preview(image_path: Path, image: CvatImage, mapping: ClassMapping, target: Path) -> None:
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        return
    for box in image.boxes:
        if mapping.class_id(box.label) is None:
            continue
        converted = convert_box(box, image.width, image.height)
        if converted is None:
            continue
        cv2.rectangle(
            frame,
            (int(max(0.0, box.xtl)), int(max(0.0, box.ytl))),
            (int(min(image.width, box.xbr)), int(min(image.height, box.ybr))),
            (0, 220, 0),
            2,
        )
        cv2.putText(
            frame,
            box.label,
            (int(max(0.0, box.xtl)), max(12, int(box.ytl) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), frame)


def convert(
    xml_paths: list[Path],
    images_root: Path,
    output_root: Path,
    mapping: ClassMapping,
    *,
    split: str = "train",
    val_fraction: float = 0.0,
    link_mode: str = "copy",
    dry_run: bool = False,
    overwrite: bool = False,
    preview_count: int = 8,
) -> ConversionStats:
    stats = ConversionStats()
    image_index = index_images(images_root)
    label_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_sources: dict[str, set[str]] = defaultdict(set)
    previews_written = 0

    seen_stems: set[str] = set()
    for xml_path in xml_paths:
        for image in parse_cvat_xml(xml_path):
            stem = Path(image.name).stem
            if stem in seen_stems:
                raise SystemExit(
                    f"image '{image.name}' appears in multiple XML files; "
                    "refusing to overwrite its label file silently"
                )
            seen_stems.add(stem)

            source_path = image_index.get(image.name)
            if source_path is None:
                stats.missing_images.append(image.name)
                continue

            target_split = assign_split(stem, split, val_fraction)
            label_lines: list[str] = []
            kept_boxes = 0
            for box in image.boxes:
                label_counts[box.label] += 1
                class_id = mapping.class_id(box.label)
                if class_id is None:
                    unknown_counts[box.label] += 1
                    unknown_sources[box.label].add(str(image.source_xml))
                    continue
                converted = convert_box(box, image.width, image.height)
                if converted is None:
                    stats.excluded_invalid_boxes += 1
                    continue
                x_center, y_center, width, height = converted
                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                kept_boxes += 1

            if not dry_run:
                label_dir = output_root / "labels" / target_split
                image_dir = output_root / "images" / target_split
                label_dir.mkdir(parents=True, exist_ok=True)
                image_dir.mkdir(parents=True, exist_ok=True)
                label_path = label_dir / f"{stem}.txt"
                image_target = image_dir / image.name
                if not overwrite and (label_path.exists() or image_target.exists()):
                    raise SystemExit(
                        f"output already exists for '{image.name}' (use --overwrite to replace)"
                    )
                label_path.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )
                _place_image(source_path, image_target, link_mode)
                if kept_boxes > 0 and previews_written < preview_count:
                    _write_preview(
                        source_path,
                        image,
                        mapping,
                        output_root / "previews" / target_split / f"{stem}.jpg",
                    )
                    previews_written += 1

            stats.converted_images += 1
            stats.converted_boxes += kept_boxes
            if kept_boxes == 0:
                stats.empty_images += 1
            stats.split_counts[target_split] = stats.split_counts.get(target_split, 0) + 1

    stats.source_class_counts = dict(label_counts)
    stats.unknown_labels = dict(unknown_counts)
    stats.unknown_label_sources = {
        label: sorted(sources) for label, sources in unknown_sources.items()
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", nargs="+", required=True, type=Path, help="CVAT XML 파일들")
    parser.add_argument("--images-root", required=True, type=Path, help="이미지 재귀 탐색 루트")
    parser.add_argument("--output-root", required=True, type=Path, help="YOLO 데이터셋 출력 루트")
    parser.add_argument("--split", default="train", help="기본 split 이름 (기본: train)")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="0보다 크면 stem 해시 기준으로 해당 비율을 'val' split으로 분리",
    )
    parser.add_argument("--class-mapping", required=True, type=Path, help="통합 클래스 JSON")
    parser.add_argument("--link-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 통계만 출력")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 덮어쓰기 허용")
    args = parser.parse_args(argv)

    for xml_path in args.xml:
        if not xml_path.is_file():
            parser.error(f"XML not found: {xml_path}")
    if not args.images_root.is_dir():
        parser.error(f"images root not found: {args.images_root}")

    mapping = ClassMapping.load(args.class_mapping)
    stats = convert(
        list(args.xml),
        args.images_root,
        args.output_root,
        mapping,
        split=args.split,
        val_fraction=args.val_fraction,
        link_mode=args.link_mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        preview_count=args.preview_count,
    )

    summary = stats.to_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        stats_path = args.output_root / f"conversion_{args.split}.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if stats.unknown_labels:
        print(
            "WARNING: unknown labels were excluded (not mapped to any class): "
            + ", ".join(sorted(stats.unknown_labels)),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
