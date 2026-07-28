from __future__ import annotations

import pytest

from vision_agent.scene_state import SceneStateStore


def test_snapshot_returns_none_for_unknown_session() -> None:
    store = SceneStateStore()

    assert store.snapshot("missing") is None
    assert store.known("missing") is False


def test_registered_session_has_empty_snapshot_without_analysis() -> None:
    store = SceneStateStore()
    store.register("session-a")

    snapshot = store.snapshot("session-a")

    assert snapshot is not None
    assert snapshot.has_analysis is False
    assert snapshot.to_dict() == {
        "visible_objects": [],
        "recent_events": [],
        "latest_narrations": [],
        "updated_at_ms": None,
    }


def test_update_keeps_most_recent_events_and_narrations() -> None:
    store = SceneStateStore(max_recent_events=2, max_narrations=2)

    store.update(
        "session-a",
        analysis_events=[{"event_type": "A"}, {"event_type": "B"}, {"event_type": "C"}],
        narrations=["one", "two", "three"],
        updated_at_ms=1_000,
    )

    snapshot = store.snapshot("session-a")
    assert snapshot is not None
    assert snapshot.has_analysis is True
    assert [event["event_type"] for event in snapshot.recent_events] == ["B", "C"]
    assert snapshot.latest_narrations == ("two", "three")
    assert snapshot.updated_at_ms == 1_000


def test_update_accumulates_across_frames_and_skips_blank_narrations() -> None:
    store = SceneStateStore()
    store.update("session-a", analysis_events=[{"event_type": "A"}], updated_at_ms=1)
    store.update("session-a", narrations=["  ", "signal turned green"], updated_at_ms=2)

    snapshot = store.snapshot("session-a")

    assert snapshot is not None
    assert [event["event_type"] for event in snapshot.recent_events] == ["A"]
    assert snapshot.latest_narrations == ("signal turned green",)
    assert snapshot.updated_at_ms == 2


def test_visible_objects_replace_previous_frame() -> None:
    store = SceneStateStore()
    store.update(
        "session-a",
        visible_objects=[{"object_type": "bus"}, {"object_type": "person"}],
        updated_at_ms=1,
    )
    store.update(
        "session-a",
        visible_objects=[{"object_type": "traffic_light"}],
        updated_at_ms=2,
    )

    snapshot = store.snapshot("session-a")

    assert snapshot is not None
    assert snapshot.visible_objects == ({"object_type": "traffic_light"},)


def test_visible_objects_none_keeps_previous_and_empty_clears() -> None:
    store = SceneStateStore()
    store.update("session-a", visible_objects=[{"object_type": "bus"}], updated_at_ms=1)
    store.update("session-a", narrations=["update without objects"], updated_at_ms=2)

    kept = store.snapshot("session-a")
    assert kept is not None
    assert kept.visible_objects == ({"object_type": "bus"},)

    store.update("session-a", visible_objects=[], updated_at_ms=3)
    cleared = store.snapshot("session-a")
    assert cleared is not None
    assert cleared.visible_objects == ()


def test_update_without_timestamp_uses_current_time() -> None:
    store = SceneStateStore()
    store.update("session-a", narrations=["hello"])

    snapshot = store.snapshot("session-a")

    assert snapshot is not None
    assert snapshot.updated_at_ms is not None
    assert snapshot.updated_at_ms > 0


def test_snapshot_is_isolated_from_later_mutations() -> None:
    store = SceneStateStore()
    event = {"event_type": "A"}
    store.update("session-a", analysis_events=[event], updated_at_ms=1)

    snapshot = store.snapshot("session-a")
    event["event_type"] = "MUTATED"
    store.update("session-a", analysis_events=[{"event_type": "B"}], updated_at_ms=2)

    assert snapshot is not None
    assert [item["event_type"] for item in snapshot.recent_events] == ["A"]


def test_sessions_are_evicted_least_recently_updated() -> None:
    store = SceneStateStore(max_sessions=2)
    store.update("first", narrations=["1"], updated_at_ms=1)
    store.update("second", narrations=["2"], updated_at_ms=2)
    store.update("first", narrations=["1b"], updated_at_ms=3)
    store.update("third", narrations=["3"], updated_at_ms=4)

    assert store.known("second") is False
    assert store.known("first") is True
    assert store.known("third") is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_recent_events": 0},
        {"max_narrations": 0},
        {"max_sessions": 0},
    ],
)
def test_store_rejects_invalid_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SceneStateStore(**kwargs)
