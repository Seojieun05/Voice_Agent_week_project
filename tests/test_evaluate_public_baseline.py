from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_public_baseline import main
from vision_agent.public_baseline.evaluation import (
    EvaluationError,
    evaluate_public_baseline,
    load_manifest,
)


def _canonical_annotation(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.setdefault("schema_version", "1.0")
    metadata = dict(result.get("video_metadata", {}))
    metadata.setdefault("fps", None)
    metadata.setdefault("frame_count", None)
    metadata.setdefault("width", None)
    metadata.setdefault("height", None)
    result["video_metadata"] = metadata
    normalized_objects: list[dict[str, object]] = []
    for raw_object in result.get("objects", []):
        item = dict(raw_object)
        item.setdefault("visible_frame_ranges", [])
        item.setdefault("signal_state_intervals", [])
        item.setdefault("transitions", [])
        item.setdefault("motion_intervals", [])
        item.setdefault("route_number", None)
        item.setdefault("screen_stage_intervals", [])
        item.setdefault("text_annotations", [])
        normalized_objects.append(item)
    result["objects"] = normalized_objects
    result.setdefault(
        "narration_constraints",
        {
            "forbidden_phrases": [],
            "forbidden_patterns": [],
            "maximum_duplicate_events": 0,
            "notes": None,
        },
    )
    return result


def _write_json(path: Path, payload: object) -> None:
    if path.parent.name == "annotations" and isinstance(payload, dict):
        payload = _canonical_annotation(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _analysis(
    object_type: str,
    stable_id: str,
    state: str,
    *,
    uncertain: bool = False,
    attributes: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "object_type": object_type,
        "stable_id": stable_id,
        "state": state,
        "confidence": 0.9,
        "attributes": attributes or {},
        "is_uncertain": uncertain,
    }


def _row(
    frame: int,
    analyses: list[dict[str, object]],
    *,
    events: list[dict[str, object]] | None = None,
    narrations: list[str] | None = None,
) -> dict[str, object]:
    detections = [
        {
            "class_name": analysis["object_type"],
            "stable_object_key": f"{analysis['object_type']}:{analysis['stable_id']}",
            "analysis": analysis,
        }
        for analysis in analyses
    ]
    return {
        "frame_index": frame,
        "timestamp_s": frame / 10,
        "inference_ms": 5.0,
        "detections": detections,
        "analysis_results": analyses,
        "analysis_events": events or [],
        "narrations": narrations or [],
    }


def _manifest(tmp_path: Path, videos: list[dict[str, object]]) -> Path:
    path = tmp_path / "dataset" / "manifest.json"
    _write_json(path, {"schema_version": "1.0", "dataset_id": "synthetic", "videos": videos})
    return path


def _run_summary(predictions: Path, video_ids: list[str]) -> None:
    _write_json(
        predictions / "run_summary.json",
        {
            "videos": [
                {
                    "video_id": video_id,
                    "status": "success",
                    "effective_fps": 20.0,
                    "realtime_factor": 0.5,
                    "source_fps": 10.0,
                    "output_jsonl": str(predictions / video_id / f"{video_id}_detections.jsonl"),
                }
                for video_id in video_ids
            ]
        },
    )


def test_manifest_validation_rejects_duplicate_video_ids(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [{"id": "same"}, {"id": "same"}])

    with pytest.raises(EvaluationError, match="중복 video ID"):
        load_manifest(manifest)


def test_unreviewed_annotation_is_qualitative_and_gt_metrics_are_null(
    tmp_path: Path,
) -> None:
    video_id = "unreviewed_signal"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic-light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "needs_manual_review",
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "traffic_light",
                    "signal_state_intervals": [{"start_frame": 0, "end_frame": 0, "state": "RED"}],
                }
            ],
        },
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [_row(0, [_analysis("traffic_light", "stable-1", "RED")])],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = json.loads(Path(summary["videos"][0]["report_json"]).read_text(encoding="utf-8"))

    assert summary["quantitatively_evaluated_video_count"] == 0
    assert report["quantitative_evaluation"] is False
    assert report["metrics"]["processed_frame_count"] == 1
    assert report["metrics"]["signal_state_frame_accuracy"] is None
    assert report["metric_reasons"]["signal_state_frame_accuracy"] == "annotation_not_reviewed"


def test_signal_metrics_count_false_green_duplicate_and_subtype_wording(
    tmp_path: Path,
) -> None:
    video_id = "reviewed_signal"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic-light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "video_metadata": {"fps": 10.0},
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "pedestrian_signal",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 3}],
                    "signal_state_intervals": [
                        {"start_frame": 0, "end_frame": 1, "state": "RED"},
                        {"start_frame": 2, "end_frame": 3, "state": "GREEN"},
                    ],
                    "transitions": [
                        {
                            "frame": 2,
                            "from_state": "RED",
                            "to_state": "GREEN",
                            "ambiguous": False,
                        }
                    ],
                }
            ],
        },
    )
    transition = {
        "event_type": "OBJECT_STATE_CHANGED",
        "object_type": "traffic_light",
        "stable_id": "stable-1",
        "previous_state": "RED",
        "current_state": "GREEN",
        "confidence": 0.9,
        "attributes": {"signal_type": "UNKNOWN"},
    }
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(0, [_analysis("traffic_light", "stable-1", "RED")]),
            _row(1, [_analysis("traffic_light", "stable-1", "RED")]),
            _row(
                2,
                [
                    _analysis(
                        "traffic_light",
                        "stable-1",
                        "GREEN",
                        attributes={
                            "signal_type": "UNKNOWN",
                            "signal_type_is_uncertain": True,
                        },
                    )
                ],
                events=[transition],
                narrations=["보행자 신호가 초록색으로 바뀌었습니다."],
            ),
            _row(
                3,
                [
                    _analysis(
                        "traffic_light",
                        "stable-1",
                        "GREEN",
                        attributes={
                            "signal_type": "PEDESTRIAN",
                            "signal_type_is_uncertain": False,
                        },
                    )
                ],
                events=[transition],
                narrations=["보행자 신호가 초록색으로 바뀌었습니다."],
            ),
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = json.loads(Path(summary["videos"][0]["report_json"]).read_text(encoding="utf-8"))
    metrics = report["metrics"]

    assert metrics["signal_state_frame_accuracy"] == 1.0
    assert metrics["signal_state_accuracy_by_state"] == {
        "RED": 1.0,
        "GREEN": 1.0,
        "YELLOW": None,
        "OFF": None,
        "UNKNOWN": None,
    }
    assert metrics["signal_confusion_matrix"]["GREEN"]["GREEN"] == 2
    assert metrics["transition_precision"] == 0.5
    assert metrics["transition_recall"] == 1.0
    assert metrics["false_green_confirmation_count"] == 0
    assert metrics["duplicate_transition_count"] == 1
    assert metrics["unconfirmed_pedestrian_signal_narration_count"] == 1


def test_bus_route_exact_wrong_number_delay_and_duplicate_approach(tmp_path: Path) -> None:
    video_id = "reviewed_bus"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "bus",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "video_metadata": {"fps": 10.0},
            "objects": [
                {
                    "ground_truth_id": "bus-1",
                    "object_type": "bus",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 2}],
                    "motion_intervals": [
                        {"start_frame": 0, "end_frame": 2, "state": "APPROACHING"}
                    ],
                    "route_number": "532",
                }
            ],
        },
    )
    approach = {
        "event_type": "OBJECT_APPROACHING",
        "object_type": "bus",
        "stable_id": "stable-1",
        "confidence": 0.9,
        "attributes": {},
    }
    approach_without_id = {
        "event_type": "OBJECT_APPROACHING",
        "object_type": "bus",
    }
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(0, [_analysis("bus", "stable-1", "APPROACHING")], events=[approach]),
            _row(
                1,
                [_analysis("bus", "stable-1", "APPROACHING")],
                events=[
                    approach,
                    {
                        "event_type": "TEXT_CONFIRMED",
                        "object_type": "bus",
                        "stable_id": "stable-1",
                        "attributes": {"route_number": "532"},
                    },
                ],
            ),
            _row(
                2,
                [_analysis("bus", "stable-1", "APPROACHING")],
                events=[
                    approach_without_id,
                    approach_without_id,
                    {
                        "event_type": "TEXT_CONFIRMED",
                        "object_type": "bus",
                        "stable_id": "stable-1",
                        "attributes": {"route_number": "999"},
                    },
                ],
            ),
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = json.loads(Path(summary["videos"][0]["report_json"]).read_text(encoding="utf-8"))
    metrics = report["metrics"]

    assert metrics["bus_detection_frame_ratio"] == 1.0
    assert metrics["bus_track_fragmentation_count"] == 0
    assert metrics["bus_approaching_precision"] == 1.0
    assert metrics["bus_approaching_recall"] == 1.0
    assert metrics["duplicate_bus_approach_event_count"] == 1
    assert metrics["route_number_exact_match"] == 1.0
    assert metrics["wrong_confirmed_route_number_count"] == 1
    assert metrics["route_number_confirmation_delay_frames"] == 1
    assert metrics["route_number_confirmation_delay_ms"] == 100.0


def test_defective_screen_false_stage_and_missing_ocr_ground_truth(tmp_path: Path) -> None:
    video_id = "ticket_machine_defective_screen"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "ticket-machine",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "objects": [
                {
                    "ground_truth_id": "machine-1",
                    "object_type": "ticket_machine",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 0}],
                    "screen_stage_intervals": [
                        {"start_frame": 0, "end_frame": 0, "stage": "UNKNOWN"}
                    ],
                    "text_annotations": [],
                }
            ],
        },
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [_row(0, [_analysis("kiosk", "stable-1", "PAYMENT")])],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = json.loads(Path(summary["videos"][0]["report_json"]).read_text(encoding="utf-8"))

    assert report["metrics"]["screen_stage_accuracy"] == 0.0
    assert report["metrics"]["defective_screen_false_stage_count"] == 1
    assert report["metrics"]["ocr_exact_match"] is None
    assert report["metric_reasons"]["ocr_exact_match"] == "ocr_text_ground_truth_not_available"
    assert report["safety_failure_count"] == 1


def test_summary_json_csv_fields_match_and_cli_writes_video_reports(
    tmp_path: Path,
) -> None:
    video_id = "qualitative"
    manifest = _manifest(tmp_path, [{"id": video_id, "category": "unknown"}])
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [_row(0, [_analysis("person", "stable-1", "UNKNOWN", uncertain=True)])],
    )
    output = tmp_path / "evaluation"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with (output / "summary.csv").open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert set(summary["videos"][0]) == set(csv_rows[0])
    assert Path(summary["videos"][0]["report_json"]).is_file()
    assert Path(summary["videos"][0]["report_markdown"]).is_file()


def _report_for(summary: dict[str, object], video_id: str) -> dict[str, object]:
    videos = summary["videos"]
    assert isinstance(videos, list)
    row = next(item for item in videos if item["video_id"] == video_id)
    return json.loads(Path(row["report_json"]).read_text(encoding="utf-8"))


def test_routed_analyzers_prefer_batch_runner_class_counts(tmp_path: Path) -> None:
    video_id = "routing_counts"
    manifest = _manifest(tmp_path, [{"id": video_id, "category": "unknown"}])
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    run_summary_path = predictions / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary["videos"][0]["inferred_analyzer_routing_counts"] = {
        "BusAnalyzer": 2,
        "GenericVisionAnalyzer": 0,
    }
    _write_json(run_summary_path, run_summary)
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [_row(0, [_analysis("person", "stable-1", "UNKNOWN", uncertain=True)])],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = _report_for(summary, video_id)

    assert report["metrics"]["routed_analyzer_types"] == ["BusAnalyzer"]


def test_missing_or_failed_prediction_has_null_metrics_and_unknown_safety(
    tmp_path: Path,
) -> None:
    missing_id = "signal_yellow_flicker_vertical"
    failed_id = "bus_london_pulls_in"
    manifest = _manifest(
        tmp_path,
        [
            {"id": missing_id, "category": "traffic_light"},
            {"id": failed_id, "category": "bus"},
        ],
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [missing_id, failed_id])
    failed_jsonl = predictions / failed_id / f"{failed_id}_detections.jsonl"
    _write_jsonl(
        failed_jsonl,
        [_row(0, [_analysis("bus", "stable-1", "APPROACHING")])],
    )
    run_summary_path = predictions / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    failed_row = next(row for row in run_summary["videos"] if row["video_id"] == failed_id)
    failed_row["status"] = "failed"
    failed_row["output_jsonl"] = str(failed_jsonl)
    _write_json(run_summary_path, run_summary)

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")

    assert summary["quantitatively_evaluated_video_count"] == 0
    for video_id in (missing_id, failed_id):
        report = _report_for(summary, video_id)
        assert report["quantitative_evaluation"] is False
        assert report["qualitative"]["prediction_available"] is False
        assert report["prediction_jsonl"] is None
        for metric_name in (
            "processed_frame_count",
            "effective_fps",
            "event_count",
            "generated_narrations",
            "routed_analyzer_types",
        ):
            assert report["metrics"][metric_name] is None
            assert report["metric_reasons"][metric_name] == "prediction_not_available"
        assert report["safety_constraints"] == [
            {
                "name": "prediction_based_safety_constraints",
                "status": "NOT_EVALUATED",
                "observed": None,
                "maximum": None,
                "reason": "prediction_not_available",
            }
        ]


def test_false_green_counts_contiguous_episodes_and_transition_cycle_by_stable_id(
    tmp_path: Path,
) -> None:
    video_id = "false_green_episodes"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic_light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "video_metadata": {"fps": 10.0},
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "traffic_light",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 4}],
                    "signal_state_intervals": [{"start_frame": 0, "end_frame": 4, "state": "RED"}],
                    "transitions": [],
                }
            ],
        },
    )

    def transition(previous: str, current: str) -> dict[str, object]:
        return {
            "event_type": "OBJECT_STATE_CHANGED",
            "object_type": "traffic_light",
            "stable_id": "stable-1",
            "previous_state": previous,
            "current_state": current,
        }

    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(
                0,
                [_analysis("traffic_light", "stable-1", "GREEN")],
                events=[transition("RED", "GREEN")],
            ),
            _row(
                1,
                [_analysis("traffic_light", "stable-1", "GREEN")],
                events=[transition("GREEN", "RED")],
            ),
            _row(
                2,
                [_analysis("traffic_light", "stable-1", "GREEN", uncertain=True)],
                events=[transition("RED", "GREEN")],
            ),
            _row(3, [_analysis("traffic_light", "stable-1", "GREEN")]),
            _row(4, [_analysis("traffic_light", "stable-1", "RED")]),
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    metrics = _report_for(summary, video_id)["metrics"]

    assert metrics["false_green_confirmation_count"] == 2
    assert metrics["duplicate_transition_count"] == 0
    assert metrics["signal_state_accuracy_by_state"] == {
        "RED": 0.2,
        "GREEN": None,
        "YELLOW": None,
        "OFF": None,
        "UNKNOWN": None,
    }


def test_kiosk_ocr_exact_match_rejects_extra_text_and_ignores_ambiguous_gt(
    tmp_path: Path,
) -> None:
    video_id = "kiosk_ocr_set"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "kiosk_like_machine",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "objects": [
                {
                    "ground_truth_id": "kiosk-1",
                    "object_type": "kiosk",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 1}],
                    "text_annotations": [
                        {"start_frame": 0, "end_frame": 1, "text": "Pay now", "ambiguous": False},
                        {"start_frame": 0, "end_frame": 1, "text": "Ignore", "ambiguous": True},
                        {"start_frame": 0, "end_frame": 1, "text": None, "ambiguous": False},
                    ],
                }
            ],
        },
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    prediction_path = predictions / video_id / f"{video_id}_detections.jsonl"
    pay_event = {
        "event_type": "TEXT_CONFIRMED",
        "object_type": "kiosk",
        "stable_id": "stable-1",
        "attributes": {"text": "PAY NOW"},
    }
    sign_hallucination = {
        "event_type": "TEXT_CONFIRMED",
        "object_type": "sign",
        "stable_id": "stable-sign",
        "attributes": {"text": "not kiosk text"},
    }
    _write_jsonl(
        prediction_path,
        [
            _row(
                0,
                [_analysis("kiosk", "stable-1", "UNKNOWN")],
                events=[pay_event, sign_hallucination],
            )
        ],
    )

    exact_summary = evaluate_public_baseline(manifest, predictions, tmp_path / "exact")
    assert _report_for(exact_summary, video_id)["metrics"]["ocr_exact_match"] == 1.0

    extra_event = {
        "event_type": "TEXT_CONFIRMED",
        "object_type": "kiosk",
        "stable_id": "stable-1",
        "attributes": {"text": "invented option"},
    }
    _write_jsonl(
        prediction_path,
        [
            _row(0, [_analysis("kiosk", "stable-1", "UNKNOWN")], events=[pay_event]),
            _row(1, [_analysis("kiosk", "stable-1", "UNKNOWN")], events=[extra_event]),
        ],
    )
    extra_summary = evaluate_public_baseline(manifest, predictions, tmp_path / "extra")

    assert _report_for(extra_summary, video_id)["metrics"]["ocr_exact_match"] == 0.0


def test_multiple_bus_routes_require_track_level_ground_truth_association(
    tmp_path: Path,
) -> None:
    video_id = "multiple_bus_routes"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "bus",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    objects = [
        {
            "ground_truth_id": f"bus-{index}",
            "object_type": "bus",
            "visible_frame_ranges": [{"start_frame": 0, "end_frame": 1}],
            "motion_intervals": [{"start_frame": 0, "end_frame": 1, "state": "APPROACHING"}],
            "route_number": route,
        }
        for index, route in enumerate(("10", "20"), start=1)
    ]
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "objects": objects,
        },
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    route_events = [
        {
            "event_type": "TEXT_CONFIRMED",
            "object_type": "bus",
            "stable_id": f"stable-{index}",
            "attributes": {"route_number": route},
        }
        for index, route in enumerate(("10", "20", "999"), start=1)
    ]
    approach_event = {
        "event_type": "OBJECT_APPROACHING",
        "object_type": "bus",
        "stable_id": "stable-1",
    }
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(
                0,
                [_analysis("bus", "stable-1", "APPROACHING")],
                events=[approach_event, approach_event, *route_events],
            )
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = _report_for(summary, video_id)

    association_metrics = (
        "bus_approaching_precision",
        "bus_approaching_recall",
        "route_number_exact_match",
        "wrong_confirmed_route_number_count",
        "route_number_confirmation_delay_frames",
        "route_number_confirmation_delay_ms",
    )
    for metric_name in association_metrics:
        assert report["metrics"][metric_name] is None
        assert (
            report["metric_reasons"][metric_name]
            == "multiple_objects_require_prediction_association"
        )
    assert report["metrics"]["duplicate_bus_approach_event_count"] == 1


def test_bus_waiting_allows_same_narration_for_different_stable_ids(tmp_path: Path) -> None:
    video_id = "bus_waiting_multiple_arrivals"
    manifest = _manifest(tmp_path, [{"id": video_id, "category": "bus"}])
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    rows: list[dict[str, object]] = []
    for frame, stable_id in enumerate(("stable-1", "stable-2")):
        event = {
            "event_type": "OBJECT_APPROACHING",
            "object_type": "bus",
            "stable_id": stable_id,
        }
        rows.append(
            _row(
                frame,
                [_analysis("bus", stable_id, "APPROACHING", attributes={"route_number": None})],
                events=[event],
                narrations=["버스가 접근하고 있습니다."],
            )
        )
    approach_without_id = {
        "event_type": "OBJECT_APPROACHING",
        "object_type": "bus",
    }
    for frame in (2, 3):
        rows.append(
            _row(
                frame,
                [_analysis("bus", f"analysis-{frame}", "UNKNOWN")],
                events=[approach_without_id],
            )
        )
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        rows,
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = _report_for(summary, video_id)
    safety = report["safety_constraints"]
    constraint = next(
        item for item in safety if item["name"] == "duplicate_bus_approach_event_limit"
    )

    assert constraint["status"] == "PASS"
    assert constraint["observed"] == 0
    assert report["qualitative"]["confirmed_route_numbers"] == []


def test_invalid_annotation_is_not_scored_as_reviewed(tmp_path: Path) -> None:
    video_id = "invalid_reviewed"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic_light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    annotation_path = manifest.parent / "annotations" / f"{video_id}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps({"video_id": video_id, "review_status": "reviewed"}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [_row(0, [_analysis("traffic_light", "stable-1", "GREEN")])],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = _report_for(summary, video_id)

    assert report["status"] == "evaluation_failed"
    assert report["quantitative_evaluation"] is False
    assert report["metrics"]["signal_state_frame_accuracy"] is None
    assert report["metric_reasons"]["signal_state_frame_accuracy"] == "video_evaluation_failed"


def test_overlapping_signal_objects_require_prediction_association(
    tmp_path: Path,
) -> None:
    video_id = "overlapping_signals"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic_light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "video_metadata": {"fps": 10.0},
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "traffic_light",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 1}],
                    "signal_state_intervals": [
                        {"start_frame": 0, "end_frame": 0, "state": "RED"},
                        {"start_frame": 1, "end_frame": 1, "state": "GREEN"},
                    ],
                    "transitions": [
                        {
                            "frame": 1,
                            "from_state": "RED",
                            "to_state": "GREEN",
                            "ambiguous": False,
                        }
                    ],
                },
                {
                    "ground_truth_id": "signal-2",
                    "object_type": "pedestrian_signal",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 1}],
                    "signal_state_intervals": [
                        {"start_frame": 0, "end_frame": 0, "state": "GREEN"},
                        {"start_frame": 1, "end_frame": 1, "state": "RED"},
                    ],
                    "transitions": [
                        {
                            "frame": 1,
                            "from_state": "GREEN",
                            "to_state": "RED",
                            "ambiguous": False,
                        }
                    ],
                },
            ],
        },
    )
    transition = {
        "event_type": "OBJECT_STATE_CHANGED",
        "object_type": "traffic_light",
        "stable_id": "predicted-1",
        "previous_state": "RED",
        "current_state": "GREEN",
    }
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(
                0,
                [_analysis("traffic_light", "predicted-1", "GREEN")],
                events=[transition],
                narrations=["보행자 신호가 초록색으로 바뀌었습니다."],
            ),
            _row(
                1,
                [_analysis("traffic_light", "predicted-1", "GREEN")],
                events=[transition],
            ),
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    report = _report_for(summary, video_id)
    association_metrics = (
        "signal_state_frame_accuracy",
        "signal_state_accuracy_by_state",
        "signal_confusion_matrix",
        "transition_precision",
        "transition_recall",
        "transition_delay_frames",
        "transition_delay_ms",
        "transition_delay_details",
        "false_green_confirmation_count",
        "unconfirmed_pedestrian_signal_narration_count",
    )
    for metric_name in association_metrics:
        assert report["metrics"][metric_name] is None
        assert (
            report["metric_reasons"][metric_name]
            == "multiple_objects_require_prediction_association"
        )
    assert report["metrics"]["duplicate_transition_count"] == 1


def test_legacy_uncertain_signal_detection_falls_back_to_unknown(
    tmp_path: Path,
) -> None:
    video_id = "legacy_uncertain_signal"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic_light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "traffic_light",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 0}],
                    "signal_state_intervals": [
                        {"start_frame": 0, "end_frame": 0, "state": "UNKNOWN"}
                    ],
                }
            ],
        },
    )
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            {
                "frame_index": 0,
                "timestamp_s": 0.0,
                "inference_ms": 5.0,
                "detections": [
                    {
                        "class_name": "traffic_light",
                        "signal_state": "GREEN",
                        "signal_state_confidence": 0.99,
                        "signal_state_is_uncertain": True,
                    }
                ],
                "analysis_results": [],
                "analysis_events": [],
                "narrations": [],
            }
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    metrics = _report_for(summary, video_id)["metrics"]

    assert metrics["signal_state_frame_accuracy"] == 1.0
    assert metrics["signal_state_accuracy_by_state"]["UNKNOWN"] == 1.0


def test_signal_transitions_without_stable_id_are_not_merged_as_duplicates(
    tmp_path: Path,
) -> None:
    video_id = "transitions_without_stable_id"
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": video_id,
                "category": "traffic_light",
                "annotation_path": f"annotations/{video_id}.json",
            }
        ],
    )
    _write_json(
        manifest.parent / "annotations" / f"{video_id}.json",
        {
            "video_id": video_id,
            "review_status": "reviewed",
            "objects": [
                {
                    "ground_truth_id": "signal-1",
                    "object_type": "traffic_light",
                    "visible_frame_ranges": [{"start_frame": 0, "end_frame": 1}],
                    "signal_state_intervals": [{"start_frame": 0, "end_frame": 1, "state": "RED"}],
                }
            ],
        },
    )
    transition_without_id = {
        "event_type": "OBJECT_STATE_CHANGED",
        "object_type": "traffic_light",
        "previous_state": "RED",
        "current_state": "GREEN",
    }
    predictions = tmp_path / "predictions"
    _run_summary(predictions, [video_id])
    _write_jsonl(
        predictions / video_id / f"{video_id}_detections.jsonl",
        [
            _row(
                frame,
                [_analysis("traffic_light", f"analysis-{frame}", "RED")],
                events=[transition_without_id],
            )
            for frame in range(2)
        ],
    )

    summary = evaluate_public_baseline(manifest, predictions, tmp_path / "evaluation")
    metrics = _report_for(summary, video_id)["metrics"]

    assert metrics["duplicate_transition_count"] == 0


def test_cli_rejects_negative_transition_tolerance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest(tmp_path, [])
    output = tmp_path / "evaluation"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--predictions",
            str(tmp_path / "predictions"),
            "--output-dir",
            str(output),
            "--transition-tolerance-frames",
            "-1",
        ]
    )

    assert exit_code == 1
    assert "0 이상" in capsys.readouterr().err
    assert not output.exists()
