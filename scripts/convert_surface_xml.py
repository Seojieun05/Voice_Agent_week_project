"""Convert AIHub sidewalk *surface* annotations into YOLO segmentation format.

The surface archives unpack into one folder per scene, each holding the frame
images, a CVAT ``annotations`` XML, and a ``MASK`` folder of rendered PNGs::

    <root>/<scene>/MP_SEL_SUR_000001.jpg
    <root>/<scene>/....xml
    <root>/<scene>/MASK/MP_SEL_SUR_000001.png

Two archives can be unpacked into the same ``--roots`` tree; every scene is
walked recursively and merged into one dataset. Output names are prefixed with
the scene path, so identical frame filenames across archives never collide:

    output_root/images/<split>/<scene>__<image>
    output_root/labels/<split>/<scene>__<stem>.txt   (polygons, normalized)
    output_root/previews/<split>/<scene>__<stem>.jpg
    output_root/conversion_surface.json

Polygons come from the XML by default (``--source xml``). ``--source mask``
traces contours out of the MASK PNGs instead and needs ``--mask-palette``.
Run ``--inspect`` first: it reports the scene layout, the XML label and element
counts, and the colors present in the MASK PNGs, without writing anything.

Unknown labels are never silently mapped; they are excluded and reported.
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
MASK_DIR_NAMES = {"mask", "masks"}
# Contours below this pixel area are annotation noise, not a walkable surface.
MIN_POLYGON_AREA_PX = 200.0


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
class CvatPolygon:
    label: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CvatImage:
    name: str
    width: int
    height: int
    polygons: tuple[CvatPolygon, ...]
    source_xml: Path


@dataclass
class ConversionStats:
    scenes: int = 0
    converted_images: int = 0
    converted_polygons: int = 0
    empty_images: int = 0
    missing_images: list[str] = field(default_factory=list)
    excluded_invalid_polygons: int = 0
    unknown_labels: dict[str, int] = field(default_factory=dict)
    unknown_label_sources: dict[str, list[str]] = field(default_factory=dict)
    source_class_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    split_scenes: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenes": self.scenes,
            "converted_images": self.converted_images,
            "converted_polygons": self.converted_polygons,
            "empty_images": self.empty_images,
            "missing_images": sorted(self.missing_images),
            "missing_image_count": len(self.missing_images),
            "excluded_invalid_polygons": self.excluded_invalid_polygons,
            "unknown_labels": dict(sorted(self.unknown_labels.items())),
            "unknown_label_sources": {
                label: sorted(paths) for label, paths in sorted(self.unknown_label_sources.items())
            },
            "source_class_counts": dict(sorted(self.source_class_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
            "split_scenes": {
                split: sorted(scenes) for split, scenes in sorted(self.split_scenes.items())
            },
        }


def scene_id(xml_path: Path, root: Path) -> str:
    """Stable, filesystem-safe id for the scene folder holding ``xml_path``."""
    relative = xml_path.parent.relative_to(root)
    parts = (root.name, *relative.parts) if relative.parts else (root.name,)
    joined = "_".join(parts)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in joined)


def parse_points(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse a CVAT ``points="x,y;x,y;..."`` attribute."""
    points: list[tuple[float, float]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_text, _, y_text = chunk.partition(",")
        points.append((float(x_text), float(y_text)))
    return tuple(points)


def parse_cvat_xml(xml_path: Path) -> tuple[list[CvatImage], Counter[str]]:
    """Parse ``.//image`` elements; also count element kinds for --inspect."""
    root = ET.parse(xml_path).getroot()
    images: list[CvatImage] = []
    element_kinds: Counter[str] = Counter()
    for element in root.findall(".//image"):
        name = element.get("name")
        width = element.get("width")
        height = element.get("height")
        if not name or width is None or height is None:
            raise ValueError(f"{xml_path}: image element missing name/width/height")
        polygons = []
        for child in element:
            element_kinds[child.tag] += 1
            if child.tag != "polygon":
                continue
            label = child.get("label")
            if label is None:
                raise ValueError(f"{xml_path}: polygon without label in image '{name}'")
            polygons.append(CvatPolygon(label=label, points=parse_points(child.get("points", ""))))
        images.append(
            CvatImage(
                name=Path(name).name,
                width=int(width),
                height=int(height),
                polygons=tuple(polygons),
                source_xml=xml_path,
            )
        )
    return images, element_kinds


def convert_polygon(
    polygon: CvatPolygon,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """Clamp to the image and return a flat normalized point list, or None."""
    if len(polygon.points) < 3:
        return None
    flat: list[float] = []
    for x, y in polygon.points:
        flat.append(min(max(x, 0.0), float(image_width)) / image_width)
        flat.append(min(max(y, 0.0), float(image_height)) / image_height)
    xs = flat[0::2]
    ys = flat[1::2]
    # Degenerate strips (a collapsed polygon) carry no area to learn from.
    if (max(xs) - min(xs)) * image_width < 2.0 or (max(ys) - min(ys)) * image_height < 2.0:
        return None
    return flat


def polygons_from_mask(
    mask_path: Path,
    palette: dict[str, tuple[int, int, int]],
) -> tuple[list[CvatPolygon], int, int]:
    """Trace one contour set per palette color out of a rendered MASK PNG."""
    import cv2
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        raise SystemExit(f"could not read mask image: {mask_path}")
    height, width = mask.shape[:2]
    polygons: list[CvatPolygon] = []
    for label, (red, green, blue) in palette.items():
        selected = cv2.inRange(mask, np.array([blue, green, red]), np.array([blue, green, red]))
        if not selected.any():
            continue
        contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_POLYGON_AREA_PX:
                continue
            epsilon = 0.002 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue
            points = tuple((float(point[0][0]), float(point[0][1])) for point in simplified)
            polygons.append(CvatPolygon(label=label, points=points))
    return polygons, width, height


def load_palette(path: Path) -> dict[str, tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    palette: dict[str, tuple[int, int, int]] = {}
    for label, color in dict(payload.get("palette", payload)).items():
        rgb = tuple(int(channel) for channel in color)
        if len(rgb) != 3:
            raise ValueError(f"{path}: color for '{label}' must be [R, G, B]")
        palette[str(label)] = rgb  # type: ignore[assignment]
    if not palette:
        raise ValueError(f"{path}: palette is empty")
    return palette


def index_scene_images(scene_dir: Path) -> dict[str, Path]:
    """Index frame images inside one scene folder, skipping MASK folders."""
    found: dict[str, Path] = {}
    for path in scene_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part.lower() in MASK_DIR_NAMES for part in path.relative_to(scene_dir).parts):
            continue
        found.setdefault(path.name, path)
    return found


def find_mask_path(scene_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for mask_dir in scene_dir.rglob("*"):
        if mask_dir.is_dir() and mask_dir.name.lower() in MASK_DIR_NAMES:
            for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
                candidate = mask_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
    return None


def assign_split(scene: str, base_split: str, val_fraction: float) -> str:
    """Route a whole scene to '<base_split>' or 'val'.

    Splitting by scene rather than by frame matters here: consecutive frames of
    one walk are near-duplicates, so a per-frame split leaks train content into
    val and reports an mAP that will not hold up on real video.
    """
    if val_fraction <= 0.0:
        return base_split
    bucket = zlib.crc32(scene.encode("utf-8")) % 1000
    return "val" if bucket < int(val_fraction * 1000) else base_split


def _place_image(source: Path, target: Path, link_mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if link_mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def _write_preview(
    image_path: Path,
    image: CvatImage,
    mapping: ClassMapping,
    target: Path,
) -> None:
    import cv2
    import numpy as np

    frame = cv2.imread(str(image_path))
    if frame is None:
        return
    overlay = frame.copy()
    for polygon in image.polygons:
        class_id = mapping.class_id(polygon.label)
        if class_id is None or len(polygon.points) < 3:
            continue
        contour = np.array([[int(x), int(y)] for x, y in polygon.points], dtype=np.int32)
        color = (
            int(37 * (class_id + 1) % 256),
            int(97 * (class_id + 1) % 256),
            int(181 * (class_id + 1) % 256),
        )
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(frame, [contour], True, color, 2)
        cv2.putText(
            frame,
            polygon.label,
            (int(contour[:, 0].min()), max(12, int(contour[:, 1].min()) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    blended = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), blended)


def collect_scenes(roots: list[Path]) -> list[tuple[str, Path, Path]]:
    """Return (scene id, scene dir, xml path) for every annotated scene."""
    scenes: list[tuple[str, Path, Path]] = []
    seen_ids: dict[str, Path] = {}
    for root in roots:
        for xml_path in sorted(root.rglob("*.xml")):
            scene = scene_id(xml_path, root)
            previous = seen_ids.get(scene)
            if previous is not None:
                raise SystemExit(
                    f"scene id '{scene}' is produced by both {previous} and {xml_path}; "
                    "unpack the archives into separate folders under --roots"
                )
            seen_ids[scene] = xml_path
            scenes.append((scene, xml_path.parent, xml_path))
    return scenes


def inspect(roots: list[Path], sample_masks: int) -> int:
    """Report the layout, labels and mask colors without writing anything."""
    import cv2
    import numpy as np

    scenes = collect_scenes(roots)
    print(f"scenes with an XML: {len(scenes)}")
    label_counts: Counter[str] = Counter()
    element_counts: Counter[str] = Counter()
    image_total = 0
    missing_masks = 0
    mask_colors: Counter[tuple[int, int, int]] = Counter()
    masks_sampled = 0

    for scene, scene_dir, xml_path in scenes:
        images, kinds = parse_cvat_xml(xml_path)
        element_counts.update(kinds)
        image_total += len(images)
        for image in images:
            for polygon in image.polygons:
                label_counts[polygon.label] += 1
        if masks_sampled < sample_masks and images:
            mask_path = find_mask_path(scene_dir, images[0].name)
            if mask_path is None:
                missing_masks += 1
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
                if mask is not None:
                    flat = mask.reshape(-1, 3)
                    unique, counts = np.unique(flat, axis=0, return_counts=True)
                    for bgr, count in zip(unique, counts):
                        mask_colors[(int(bgr[2]), int(bgr[1]), int(bgr[0]))] += int(count)
                    masks_sampled += 1

    print(f"images referenced by XML: {image_total}")
    print("\nXML child elements per image:")
    for tag, count in element_counts.most_common():
        print(f"  {tag}: {count}")
    print("\npolygon labels:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    if not label_counts:
        print("  (none — the XML has no <polygon>; use --source mask)")
    print(f"\nmask PNGs sampled: {masks_sampled} (scenes with no matching mask: {missing_masks})")
    if mask_colors:
        print("mask colors (R, G, B) by pixel count:")
        for rgb, count in mask_colors.most_common(24):
            print(f"  {list(rgb)}: {count}")
    print("\nsample scene ids:")
    for scene, _, _ in scenes[:5]:
        print(f"  {scene}")
    return 0


def convert(
    roots: list[Path],
    output_root: Path,
    mapping: ClassMapping,
    *,
    source: str = "xml",
    palette: dict[str, tuple[int, int, int]] | None = None,
    split: str = "train",
    val_fraction: float = 0.0,
    link_mode: str = "copy",
    dry_run: bool = False,
    overwrite: bool = False,
    preview_count: int = 8,
) -> ConversionStats:
    stats = ConversionStats()
    label_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_sources: dict[str, set[str]] = defaultdict(set)
    split_scenes: dict[str, set[str]] = defaultdict(set)
    previews_written = 0
    seen_outputs: set[str] = set()

    scenes = collect_scenes(roots)
    stats.scenes = len(scenes)
    for scene, scene_dir, xml_path in scenes:
        images, _ = parse_cvat_xml(xml_path)
        image_index = index_scene_images(scene_dir)
        target_split = assign_split(scene, split, val_fraction)

        for image in images:
            source_path = image_index.get(image.name)
            if source_path is None:
                stats.missing_images.append(f"{scene}/{image.name}")
                continue

            if source == "mask":
                assert palette is not None
                mask_path = find_mask_path(scene_dir, image.name)
                if mask_path is None:
                    stats.missing_images.append(f"{scene}/MASK/{Path(image.name).stem}")
                    continue
                traced, width, height = polygons_from_mask(mask_path, palette)
                image = CvatImage(
                    name=image.name,
                    width=width,
                    height=height,
                    polygons=tuple(traced),
                    source_xml=mask_path,
                )

            output_stem = f"{scene}__{Path(image.name).stem}"
            if output_stem in seen_outputs:
                raise SystemExit(
                    f"image '{image.name}' appears twice in scene '{scene}'; "
                    "refusing to overwrite its label file silently"
                )
            seen_outputs.add(output_stem)

            label_lines: list[str] = []
            kept = 0
            for polygon in image.polygons:
                label_counts[polygon.label] += 1
                class_id = mapping.class_id(polygon.label)
                if class_id is None:
                    unknown_counts[polygon.label] += 1
                    unknown_sources[polygon.label].add(str(image.source_xml))
                    continue
                converted = convert_polygon(polygon, image.width, image.height)
                if converted is None:
                    stats.excluded_invalid_polygons += 1
                    continue
                coordinates = " ".join(f"{value:.6f}" for value in converted)
                label_lines.append(f"{class_id} {coordinates}")
                kept += 1

            if not dry_run:
                label_dir = output_root / "labels" / target_split
                image_dir = output_root / "images" / target_split
                label_dir.mkdir(parents=True, exist_ok=True)
                image_dir.mkdir(parents=True, exist_ok=True)
                label_path = label_dir / f"{output_stem}.txt"
                image_target = image_dir / f"{output_stem}{source_path.suffix}"
                if not overwrite and (label_path.exists() or image_target.exists()):
                    raise SystemExit(
                        f"output already exists for '{output_stem}' (use --overwrite to replace)"
                    )
                label_path.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )
                _place_image(source_path, image_target, link_mode)
                if kept > 0 and previews_written < preview_count:
                    _write_preview(
                        source_path,
                        image,
                        mapping,
                        output_root / "previews" / target_split / f"{output_stem}.jpg",
                    )
                    previews_written += 1

            stats.converted_images += 1
            stats.converted_polygons += kept
            if kept == 0:
                stats.empty_images += 1
            stats.split_counts[target_split] = stats.split_counts.get(target_split, 0) + 1
            split_scenes[target_split].add(scene)

    stats.source_class_counts = dict(label_counts)
    stats.unknown_labels = dict(unknown_counts)
    stats.unknown_label_sources = {
        label: sorted(sources) for label, sources in unknown_sources.items()
    }
    stats.split_scenes = {split_name: sorted(names) for split_name, names in split_scenes.items()}
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        type=Path,
        help="압축 해제한 surface 데이터 루트들 (하위 폴더를 재귀 탐색)",
    )
    parser.add_argument("--output-root", type=Path, help="YOLO 데이터셋 출력 루트")
    parser.add_argument("--class-mapping", type=Path, help="통합 클래스 JSON")
    parser.add_argument(
        "--source",
        choices=("xml", "mask"),
        default="xml",
        help="폴리곤 출처: XML의 <polygon> (기본) 또는 MASK PNG 윤곽선",
    )
    parser.add_argument(
        "--mask-palette",
        type=Path,
        help="--source mask일 때 클래스 → [R, G, B] 매핑 JSON",
    )
    parser.add_argument("--split", default="train", help="기본 split 이름 (기본: train)")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="0보다 크면 scene 해시 기준으로 해당 비율을 'val' split으로 분리",
    )
    parser.add_argument("--link-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 통계만 출력")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 덮어쓰기 허용")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="변환 없이 폴더 구조/라벨/MASK 색상만 출력하고 종료",
    )
    parser.add_argument(
        "--inspect-mask-samples",
        type=int,
        default=5,
        help="--inspect에서 색상을 조사할 MASK PNG 개수 (기본: 5)",
    )
    args = parser.parse_args(argv)

    for root in args.roots:
        if not root.is_dir():
            parser.error(f"root not found: {root}")

    if args.inspect:
        return inspect(list(args.roots), args.inspect_mask_samples)

    if args.output_root is None or args.class_mapping is None:
        parser.error("--output-root and --class-mapping are required unless --inspect is used")
    palette = None
    if args.source == "mask":
        if args.mask_palette is None:
            parser.error("--source mask requires --mask-palette")
        palette = load_palette(args.mask_palette)

    mapping = ClassMapping.load(args.class_mapping)
    stats = convert(
        list(args.roots),
        args.output_root,
        mapping,
        source=args.source,
        palette=palette,
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
        stats_path = args.output_root / "conversion_surface.json"
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
