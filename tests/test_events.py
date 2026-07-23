import pytest

from vision_agent.events import StableObjectEventEngine
from vision_agent.types import Detection, SignalState


def detection(
    frame: int,
    track_id: int | None = 1,
    *,
    xyxy: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 50.0),
    class_id: int = 9,
    class_name: str = "traffic light",
) -> Detection:
    return Detection(
        frame_index=frame,
        timestamp_s=frame / 30,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        xyxy=xyxy,
        track_id=track_id,
    )


def test_appeared_after_stable_frames() -> None:
    engine = StableObjectEventEngine(min_seen_frames=3, max_missed_frames=2)
    assert engine.update([detection(0)], 0.0) == []
    assert engine.update([detection(1)], 1 / 30) == []

    events = engine.update([detection(2)], 2 / 30)
    assert len(events) == 1
    assert events[0].event_type == "appeared"
    assert events[0].object_key.startswith("traffic light:stable-")
    assert engine.update([detection(3)], 3 / 30) == []


def test_disappeared_after_missed_frames() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1, max_missed_frames=2)
    engine.update([detection(0)], 0.0)
    assert engine.update([], 1 / 30) == []

    events = engine.update([], 2 / 30)
    assert len(events) == 1
    assert events[0].event_type == "disappeared"
    assert engine.update([], 3 / 30) == []


def test_frame_update_reports_retired_transient_object_without_presence_event() -> None:
    engine = StableObjectEventEngine(min_seen_frames=3, max_missed_frames=2)
    first = engine.update_frame([detection(0)], 0.0)

    assert engine.update_frame([], 1 / 30).retired_object_keys == ()
    retired = engine.update_frame([], 2 / 30)

    assert retired.events == ()
    assert retired.retired_object_keys == first.object_keys


def test_frame_update_reports_announced_object_as_retired_with_disappearance() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1, max_missed_frames=1)
    appeared = engine.update_frame([detection(0)], 0.0)
    retired = engine.update_frame([], 1 / 30)

    assert [event.event_type for event in retired.events] == ["disappeared"]
    assert retired.retired_object_keys == appeared.object_keys


def test_temporary_miss_does_not_repeat_events() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        max_missed_frames=3,
        max_reconnect_frames=1,
    )
    appeared = engine.update([detection(0, 1)], 0.0)

    assert engine.update([], 1 / 30) == []
    assert engine.update([detection(2, 5)], 2 / 30) == []
    assert engine.update([], 3 / 30) == []
    assert engine.update([], 4 / 30) == []
    disappeared = engine.update([], 5 / 30)

    assert len(appeared) == 1
    assert len(disappeared) == 1
    assert disappeared[0].event_type == "disappeared"
    assert disappeared[0].object_key == appeared[0].object_key


def test_track_id_one_none_five_reconnects_to_one_stable_object() -> None:
    engine = StableObjectEventEngine(min_seen_frames=3, max_missed_frames=4)

    assert engine.update([detection(0, 1)], 0.0) == []
    assert engine.update([detection(1, None)], 1 / 30) == []
    events = engine.update([detection(2, 5)], 2 / 30)

    assert len(events) == 1
    assert events[0].event_type == "appeared"
    assert engine.update([detection(3, 5)], 3 / 30) == []


def test_two_untracked_objects_of_same_class_remain_separate() -> None:
    engine = StableObjectEventEngine(min_seen_frames=2, max_missed_frames=3)
    left = (0.0, 0.0, 10.0, 10.0)
    right = (100.0, 0.0, 110.0, 10.0)

    assert (
        engine.update(
            [detection(0, None, xyxy=left), detection(0, None, xyxy=right)],
            0.0,
        )
        == []
    )
    events = engine.update(
        [detection(1, None, xyxy=right), detection(1, None, xyxy=left)],
        1 / 30,
    )

    assert len(events) == 2
    assert {event.event_type for event in events} == {"appeared"}
    assert len({event.object_key for event in events}) == 2


def test_far_detection_with_new_track_id_is_a_new_object() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1, max_missed_frames=4)
    first = engine.update(
        [detection(0, 1, xyxy=(0.0, 0.0, 10.0, 10.0))],
        0.0,
    )
    second = engine.update(
        [detection(1, 5, xyxy=(100.0, 100.0, 110.0, 110.0))],
        1 / 30,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].event_type == "appeared"
    assert second[0].object_key != first[0].object_key


def test_one_detection_cannot_update_two_overlapping_states() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        max_missed_frames=1,
        reconnect_iou_threshold=0.3,
    )
    first_events = engine.update(
        [
            detection(0, 1, xyxy=(0.0, 0.0, 10.0, 10.0)),
            detection(0, 2, xyxy=(5.0, 0.0, 15.0, 10.0)),
        ],
        0.0,
    )
    events = engine.update(
        [detection(1, None, xyxy=(2.5, 0.0, 12.5, 10.0))],
        1 / 30,
    )

    assert len(first_events) == 2
    assert len(events) == 1
    assert events[0].event_type == "disappeared"
    assert events[0].object_key in {event.object_key for event in first_events}


def test_same_raw_track_id_has_priority_over_iou() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1, max_missed_frames=1)
    appeared = engine.update(
        [detection(0, 7, xyxy=(0.0, 0.0, 10.0, 10.0))],
        0.0,
    )

    assert (
        engine.update(
            [detection(1, 7, xyxy=(100.0, 100.0, 110.0, 110.0))],
            1 / 30,
        )
        == []
    )
    disappeared = engine.update([], 2 / 30)

    assert len(disappeared) == 1
    assert disappeared[0].object_key == appeared[0].object_key


def test_detection_beyond_reconnect_window_creates_a_new_object() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        max_missed_frames=4,
        max_reconnect_frames=1,
    )
    appeared = engine.update([detection(0, 1)], 0.0)
    assert engine.update([], 1 / 30) == []
    assert engine.update([], 2 / 30) == []

    reappeared = engine.update([detection(3, 5)], 3 / 30)

    assert len(reappeared) == 1
    assert reappeared[0].event_type == "appeared"
    assert reappeared[0].object_key != appeared[0].object_key


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_reconnect_iou_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="reconnect_iou_threshold"):
        StableObjectEventEngine(reconnect_iou_threshold=threshold)


def test_negative_max_reconnect_frames_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_reconnect_frames"):
        StableObjectEventEngine(max_reconnect_frames=-1)


def test_zero_reconnect_frames_still_allows_consecutive_spatial_match() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=2,
        max_missed_frames=3,
        max_reconnect_frames=0,
    )

    assert engine.update([detection(0, 1)], 0.0) == []
    events = engine.update([detection(1, 5)], 1 / 30)

    assert len(events) == 1
    assert events[0].event_type == "appeared"


@pytest.mark.parametrize("raw_track_id", [None, 5])
def test_detection_dict_preserves_raw_track_id(raw_track_id: int | None) -> None:
    assert detection(0, raw_track_id).to_dict()["track_id"] == raw_track_id


def test_initial_signal_state_becomes_baseline_without_change_event() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        min_signal_state_frames=3,
    )

    for frame in range(3):
        events = engine.update(
            [detection(frame)],
            frame / 30,
            signal_states=[SignalState.GREEN],
        )

    assert [event.event_type for event in events] == []


def test_signal_change_requires_consecutive_states_and_emits_once() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        min_signal_state_frames=3,
    )
    for frame in range(3):
        engine.update(
            [detection(frame)],
            frame / 30,
            signal_states=[SignalState.GREEN],
        )

    assert (
        engine.update(
            [detection(3)],
            3 / 30,
            signal_states=[SignalState.RED],
        )
        == []
    )
    assert (
        engine.update(
            [detection(4)],
            4 / 30,
            signal_states=[SignalState.RED],
        )
        == []
    )
    events = engine.update(
        [detection(5)],
        5 / 30,
        signal_states=[SignalState.RED],
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "signal_changed"
    assert event.previous_state is SignalState.GREEN
    assert event.current_state is SignalState.RED
    assert event.to_dict()["previous_state"] == "GREEN"
    assert event.to_dict()["current_state"] == "RED"
    assert (
        engine.update(
            [detection(6)],
            6 / 30,
            signal_states=[SignalState.RED],
        )
        == []
    )


def test_red_to_green_change_uses_third_green_timestamp() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        min_signal_state_frames=3,
    )
    for frame in range(3):
        engine.update(
            [detection(frame)],
            frame / 25,
            signal_states=[SignalState.RED],
        )

    engine.update([detection(3)], 1.08, signal_states=[SignalState.GREEN])
    engine.update([detection(4)], 1.12, signal_states=[SignalState.GREEN])
    events = engine.update(
        [detection(5)],
        1.16,
        signal_states=[SignalState.GREEN],
    )

    assert len(events) == 1
    assert events[0].previous_state is SignalState.RED
    assert events[0].current_state is SignalState.GREEN
    assert events[0].timestamp_s == 1.16


def test_unknown_and_missing_detection_break_only_candidate_streak() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        max_missed_frames=4,
        min_signal_state_frames=3,
    )
    for frame in range(3):
        engine.update(
            [detection(frame)],
            frame / 30,
            signal_states=[SignalState.GREEN],
        )

    engine.update([detection(3)], 3 / 30, signal_states=[SignalState.RED])
    engine.update([detection(4)], 4 / 30, signal_states=[SignalState.UNKNOWN])
    engine.update([detection(5)], 5 / 30, signal_states=[SignalState.RED])
    assert engine.update([], 6 / 30) == []
    assert (
        engine.update(
            [detection(7, 5)],
            7 / 30,
            signal_states=[SignalState.RED],
        )
        == []
    )
    assert (
        engine.update(
            [detection(8, 5)],
            8 / 30,
            signal_states=[SignalState.RED],
        )
        == []
    )
    events = engine.update(
        [detection(9, 5)],
        9 / 30,
        signal_states=[SignalState.RED],
    )

    assert [event.event_type for event in events] == ["signal_changed"]


def test_signal_states_are_independent_for_two_objects() -> None:
    engine = StableObjectEventEngine(
        min_seen_frames=1,
        min_signal_state_frames=2,
    )
    left = detection(0, 1, xyxy=(0.0, 0.0, 10.0, 10.0))
    right = detection(0, 2, xyxy=(100.0, 0.0, 110.0, 10.0))
    for frame in range(2):
        engine.update(
            [
                detection(frame, 1, xyxy=left.xyxy),
                detection(frame, 2, xyxy=right.xyxy),
            ],
            frame / 30,
            signal_states=[SignalState.GREEN, SignalState.RED],
        )

    engine.update(
        [detection(2, 1, xyxy=left.xyxy), detection(2, 2, xyxy=right.xyxy)],
        2 / 30,
        signal_states=[SignalState.RED, SignalState.RED],
    )
    events = engine.update(
        [detection(3, 1, xyxy=left.xyxy), detection(3, 2, xyxy=right.xyxy)],
        3 / 30,
        signal_states=[SignalState.RED, SignalState.RED],
    )

    changes = [event for event in events if event.event_type == "signal_changed"]
    assert len(changes) == 1
    assert changes[0].previous_state is SignalState.GREEN
    assert changes[0].current_state is SignalState.RED


def test_update_frame_returns_stable_keys_aligned_to_detections() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1)
    result = engine.update_frame(
        [detection(0, 1), detection(0, 2, xyxy=(100.0, 0.0, 110.0, 10.0))],
        0.0,
        signal_states=[SignalState.GREEN, SignalState.RED],
    )

    assert len(result.object_keys) == 2
    assert result.object_keys[0] != result.object_keys[1]
    assert all(key.startswith("traffic light:stable-") for key in result.object_keys)


def test_reset_forgets_tracks_and_restarts_stable_id_allocation() -> None:
    engine = StableObjectEventEngine(min_seen_frames=1, max_missed_frames=1)
    first = engine.update_frame([detection(0, 7)], 0.0)

    engine.reset()
    assert engine.update_frame([], 1 / 30).events == ()
    restarted = engine.update_frame([detection(2, 7)], 2 / 30)

    assert first.object_keys == ("traffic light:stable-1",)
    assert restarted.object_keys == ("traffic light:stable-1",)
    assert [event.event_type for event in restarted.events] == ["appeared"]


def test_signal_state_configuration_and_alignment_are_validated() -> None:
    with pytest.raises(ValueError, match="min_signal_state_frames"):
        StableObjectEventEngine(min_signal_state_frames=0)

    engine = StableObjectEventEngine()
    with pytest.raises(ValueError, match="same length"):
        engine.update([detection(0)], 0.0, signal_states=[])


def test_existing_event_json_does_not_gain_signal_fields() -> None:
    event = StableObjectEventEngine(min_seen_frames=1).update([detection(0)], 0.0)[0]

    assert set(event.to_dict()) == {
        "event_type",
        "object_key",
        "class_name",
        "timestamp_s",
        "message",
    }
