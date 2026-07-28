from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.convert_cvat_xml import (
    ClassMapping,
    CvatBox,
    assign_split,
    convert,
    convert_box,
    index_images,
    main,
    parse_cvat_xml,
)

FIXTURE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><name>fixture</name></task></meta>
  <image id="0" name="img_a.jpg" width="100" height="50">
    <box label="person" xtl="10.0" ytl="10.0" xbr="30.0" ybr="40.0" occluded="0"/>
    <box label="bollard" xtl="-5.0" ytl="20.0" xbr="20.0" ybr="60.0" occluded="0"/>
    <box label="person" xtl="40.0" ytl="30.0" xbr="35.0" ybr="45.0" occluded="0"/>
    <box label="alien_object" xtl="1.0" ytl="1.0" xbr="9.0" ybr="9.0" occluded="0"/>
  </image>
  <image id="1" name="img_empty.jpg" width="100" height="50">
  </image>
  <image id="2" name="img_missing.jpg" width="100" height="50">
    <box label="person" xtl="1.0" ytl="1.0" xbr="9.0" ybr="9.0" occluded="0"/>
  </image>
</annotations>
"""


@pytest.fixture()
def mapping() -> ClassMapping:
    return ClassMapping(classes=("person", "bollard"), label_map={})


@pytest.fixture()
def dataset(tmp_path: Path) -> dict[str, Path]:
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(FIXTURE_XML, encoding="utf-8")
    images_root = tmp_path / "raw"
    (images_root / "nested").mkdir(parents=True)
    frame = np.full((50, 100, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(images_root / "img_a.jpg"), frame)
    cv2.imwrite(str(images_root / "nested" / "img_empty.jpg"), frame)
    # img_missing.jpg is intentionally absent.
    return {
        "xml": xml_path,
        "images_root": images_root,
        "output_root": tmp_path / "yolo",
    }


def test_parse_cvat_xml_reads_images_and_boxes(dataset: dict[str, Path]) -> None:
    images = parse_cvat_xml(dataset["xml"])

    assert [image.name for image in images] == ["img_a.jpg", "img_empty.jpg", "img_missing.jpg"]
    assert images[0].width == 100
    assert images[0].height == 50
    assert [box.label for box in images[0].boxes] == [
        "person",
        "bollard",
        "person",
        "alien_object",
    ]
    assert images[1].boxes == ()


def test_convert_box_normalizes_and_clamps() -> None:
    box = CvatBox(label="person", xtl=10.0, ytl=10.0, xbr=30.0, ybr=40.0)
    assert convert_box(box, 100, 50) == pytest.approx((0.2, 0.5, 0.2, 0.6))

    clamped = convert_box(CvatBox("person", -5.0, 20.0, 20.0, 60.0), 100, 50)
    assert clamped == pytest.approx((0.1, 0.7, 0.2, 0.6))

    assert convert_box(CvatBox("person", 40.0, 30.0, 35.0, 45.0), 100, 50) is None
    assert convert_box(CvatBox("person", 10.0, 30.0, 20.0, 30.0), 100, 50) is None


def test_convert_writes_labels_previews_and_stats(
    dataset: dict[str, Path],
    mapping: ClassMapping,
) -> None:
    stats = convert(
        [dataset["xml"]],
        dataset["images_root"],
        dataset["output_root"],
        mapping,
        split="train",
    )

    label_a = dataset["output_root"] / "labels" / "train" / "img_a.txt"
    lines = label_a.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("0 0.200000 0.500000")
    assert lines[1].startswith("1 0.100000 0.700000")
    assert len(lines) == 2

    label_empty = dataset["output_root"] / "labels" / "train" / "img_empty.txt"
    assert label_empty.read_text(encoding="utf-8") == ""

    assert (dataset["output_root"] / "images" / "train" / "img_a.jpg").is_file()
    assert list((dataset["output_root"] / "previews" / "train").glob("*.jpg"))

    assert stats.converted_images == 2
    assert stats.converted_boxes == 2
    assert stats.empty_images == 1
    assert stats.missing_images == ["img_missing.jpg"]
    assert stats.excluded_invalid_boxes == 1
    assert stats.unknown_labels == {"alien_object": 1}
    assert stats.unknown_label_sources == {"alien_object": [str(dataset["xml"])]}
    # img_missing.jpg is skipped before its boxes are counted.
    assert stats.source_class_counts["person"] == 2


def test_dry_run_writes_nothing(dataset: dict[str, Path], mapping: ClassMapping) -> None:
    stats = convert(
        [dataset["xml"]],
        dataset["images_root"],
        dataset["output_root"],
        mapping,
        dry_run=True,
    )

    assert stats.converted_images == 2
    assert not dataset["output_root"].exists()


def test_existing_output_requires_overwrite(
    dataset: dict[str, Path],
    mapping: ClassMapping,
) -> None:
    convert([dataset["xml"]], dataset["images_root"], dataset["output_root"], mapping)

    with pytest.raises(SystemExit, match="--overwrite"):
        convert([dataset["xml"]], dataset["images_root"], dataset["output_root"], mapping)

    convert(
        [dataset["xml"]],
        dataset["images_root"],
        dataset["output_root"],
        mapping,
        overwrite=True,
    )


def test_symlink_mode_links_instead_of_copying(
    dataset: dict[str, Path],
    mapping: ClassMapping,
) -> None:
    convert(
        [dataset["xml"]],
        dataset["images_root"],
        dataset["output_root"],
        mapping,
        link_mode="symlink",
    )

    linked = dataset["output_root"] / "images" / "train" / "img_a.jpg"
    assert linked.is_symlink()
    assert linked.resolve() == (dataset["images_root"] / "img_a.jpg").resolve()


def test_duplicate_image_filenames_are_an_error(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a" / "dup.jpg"), frame)
    cv2.imwrite(str(tmp_path / "b" / "dup.jpg"), frame)

    with pytest.raises(SystemExit, match="duplicate image filenames"):
        index_images(tmp_path)


def test_assign_split_is_deterministic_and_splits() -> None:
    stems = [f"img_{i:04d}" for i in range(1000)]
    assignments = {stem: assign_split(stem, "train", 0.1) for stem in stems}

    assert assignments == {stem: assign_split(stem, "train", 0.1) for stem in stems}
    val_count = sum(1 for split in assignments.values() if split == "val")
    assert 50 < val_count < 200
    assert assign_split("anything", "train", 0.0) == "train"


def test_class_mapping_rejects_bad_configs(tmp_path: Path) -> None:
    bad_alias = tmp_path / "bad.json"
    bad_alias.write_text(
        json.dumps({"classes": ["person"], "label_map": {"stop": "stop_sign"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown class"):
        ClassMapping.load(bad_alias)

    duplicated = tmp_path / "dup.json"
    duplicated.write_text(json.dumps({"classes": ["person", "person"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        ClassMapping.load(duplicated)


def test_main_end_to_end_writes_stats_json(
    dataset: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"classes": ["person", "bollard"]}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--xml",
            str(dataset["xml"]),
            "--images-root",
            str(dataset["images_root"]),
            "--output-root",
            str(dataset["output_root"]),
            "--class-mapping",
            str(mapping_path),
        ]
    )

    assert exit_code == 0
    stats_path = dataset["output_root"] / "conversion_train.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["converted_images"] == 2
    assert payload["unknown_labels"] == {"alien_object": 1}
    captured = capsys.readouterr()
    assert "alien_object" in captured.err
