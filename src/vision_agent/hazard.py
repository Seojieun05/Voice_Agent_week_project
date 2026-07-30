"""Detect objects closing on the user fast enough to be a collision risk.

The existing ``OBJECT_APPROACHING`` event only fires for bus/car/vehicle and only
when an object-specific analyzer reports an ``APPROACHING`` state, which the bus
analyzer derives from route-stop behaviour. That is the wrong tool for "a bike is
about to hit me": it is class-limited and tuned for slow, deliberate motion.

This module works straight off tracked bounding boxes instead, so it covers every
class the detector knows — bicycle, scooter, motorcycle, person, and also static
obstacles the user is walking into, such as a bollard.

Time to contact from a single camera
------------------------------------
A box's area ratio ``a`` scales with the inverse square of distance ``d``::

    a ∝ 1/d²  ⇒  ln a = -2 ln d + c  ⇒  d(ln a)/dt = -2 · d'/d

With closing speed ``v = -d'`` the time to contact is ``d/v``, so::

    TTC = 2 / (d(ln a)/dt)

Only the *rate of change* of apparent size is needed — no calibration, no known
object size. The cost is sensitivity to box jitter, so a rate is only trusted
when most recent samples agree on the direction of growth.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .object_types import normalize_object_type
from .types import AnalysisEvent, Detection

HAZARD_DETECTED: Final = "HAZARD_DETECTED"


class HazardLevel(str, Enum):
    """How much time the user has left before contact."""

    WARNING = "WARNING"
    IMMINENT = "IMMINENT"


class HazardZone(str, Enum):
    """Where the closing object sits horizontally, from the user's view."""

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


@dataclass(frozen=True, slots=True)
class HazardAssessment:
    """One object judged to be closing on the user."""

    stable_id: str
    object_type: str
    level: HazardLevel
    zone: HazardZone
    time_to_contact_s: float
    growth_rate: float
    area_ratio: float
    in_path: bool
    confidence: float
    #: Counts emissions for this track. Carried into the narration deduplication
    #: key so a repeat is treated as new information rather than a duplicate —
    #: this monitor's ``repeat_interval_s`` is the only throttle for hazards.
    emission_index: int = 0


#: Classes that move on their own and can reach the user quickly. These trigger
#: at :attr:`HazardLevel.WARNING`; anything else needs the stricter
#: :attr:`HazardLevel.IMMINENT` bar, so a signpost only warns when the user is
#: genuinely about to walk into it.
DEFAULT_MOVING_HAZARD_TYPES: Final = frozenset(
    {
        "person",
        "bicycle",
        "motorcycle",
        "scooter",
        "car",
        "vehicle",
        "bus",
        "truck",
        "carrier",
        "stroller",
        "wheelchair",
    }
)


@dataclass(slots=True)
class _TrackHistory:
    samples: deque[tuple[float, float]]
    last_emitted_level: HazardLevel | None = None
    last_emitted_at_s: float | None = None
    emission_count: int = 0


class ApproachMonitor:
    """Estimate time to contact per tracked object and flag the dangerous ones.

    Stateful across frames but free of I/O and framework types, so the whole
    escalation policy is unit-testable with synthetic box sequences.
    """

    def __init__(
        self,
        *,
        moving_hazard_types: Iterable[str] = DEFAULT_MOVING_HAZARD_TYPES,
        history_samples: int = 6,
        minimum_samples: int = 3,
        minimum_span_s: float = 0.25,
        warning_ttc_s: float = 3.5,
        imminent_ttc_s: float = 1.5,
        minimum_area_ratio: float = 0.005,
        corridor_ratio: float = 0.5,
        minimum_corridor_overlap: float = 0.2,
        minimum_growth_consistency: float = 0.75,
        repeat_interval_s: float = 2.0,
        minimum_confidence: float = 0.35,
    ) -> None:
        if history_samples < 2:
            raise ValueError("history_samples must be at least 2")
        if not 2 <= minimum_samples <= history_samples:
            raise ValueError("minimum_samples must be between 2 and history_samples")
        if minimum_span_s <= 0.0:
            raise ValueError("minimum_span_s must be positive")
        if imminent_ttc_s <= 0.0 or warning_ttc_s <= 0.0:
            raise ValueError("time-to-contact thresholds must be positive")
        if imminent_ttc_s > warning_ttc_s:
            raise ValueError("imminent_ttc_s must not exceed warning_ttc_s")
        if not 0.0 <= minimum_area_ratio < 1.0:
            raise ValueError("minimum_area_ratio must be in [0, 1)")
        if not 0.0 < corridor_ratio <= 1.0:
            raise ValueError("corridor_ratio must be in (0, 1]")
        if not 0.0 <= minimum_corridor_overlap <= 1.0:
            raise ValueError("minimum_corridor_overlap must be in [0, 1]")
        if not 0.0 <= minimum_growth_consistency <= 1.0:
            raise ValueError("minimum_growth_consistency must be in [0, 1]")
        if repeat_interval_s < 0.0:
            raise ValueError("repeat_interval_s must be non-negative")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")

        self.moving_hazard_types = frozenset(
            normalize_object_type(name) for name in moving_hazard_types
        )
        self.history_samples = history_samples
        self.minimum_samples = minimum_samples
        self.minimum_span_s = float(minimum_span_s)
        self.warning_ttc_s = float(warning_ttc_s)
        self.imminent_ttc_s = float(imminent_ttc_s)
        self.minimum_area_ratio = float(minimum_area_ratio)
        self.corridor_ratio = float(corridor_ratio)
        self.minimum_corridor_overlap = float(minimum_corridor_overlap)
        self.minimum_growth_consistency = float(minimum_growth_consistency)
        self.repeat_interval_s = float(repeat_interval_s)
        self.minimum_confidence = float(minimum_confidence)
        self._history: dict[str, _TrackHistory] = {}

    def update(
        self,
        tracked: Sequence[tuple[str, Detection]],
        *,
        frame_width: int,
        frame_height: int,
        timestamp_s: float,
    ) -> list[HazardAssessment]:
        """Fold one frame in and return the hazards worth announcing right now.

        ``tracked`` pairs each detection with the stable object id the event
        engine assigned, so history survives raw tracker id churn.
        """
        if frame_width <= 0 or frame_height <= 0:
            return []

        assessments: list[HazardAssessment] = []
        seen: set[str] = set()
        for stable_id, detection in tracked:
            key = str(stable_id).strip()
            if not key:
                continue
            seen.add(key)
            assessment = self._observe(
                key,
                detection,
                frame_width=frame_width,
                frame_height=frame_height,
                timestamp_s=timestamp_s,
            )
            if assessment is not None:
                assessments.append(assessment)

        # An object that left the frame must not keep its stale growth history:
        # if the tracker hands the same stable id to a re-entering object, the
        # gap would read as a huge jump in apparent size.
        for stale_id in self._history.keys() - seen:
            del self._history[stale_id]

        # Most urgent first so a caller taking only one message takes the worst.
        assessments.sort(key=lambda item: item.time_to_contact_s)
        return assessments

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one object's history, or all of it at a session boundary."""
        if stable_id is None:
            self._history.clear()
            return
        self._history.pop(str(stable_id).strip(), None)

    def _observe(
        self,
        stable_id: str,
        detection: Detection,
        *,
        frame_width: int,
        frame_height: int,
        timestamp_s: float,
    ) -> HazardAssessment | None:
        left, top, right, bottom = (float(value) for value in detection.xyxy)
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        if width <= 0.0 or height <= 0.0:
            return None
        area_ratio = (width * height) / float(frame_width * frame_height)
        if area_ratio <= 0.0:
            return None

        history = self._history.setdefault(
            stable_id,
            _TrackHistory(samples=deque(maxlen=self.history_samples)),
        )
        samples = history.samples
        # Two detections sharing a timestamp would divide by zero below; keeping
        # the newer box is closer to the truth than averaging them.
        if samples and math.isclose(samples[-1][0], timestamp_s):
            samples[-1] = (timestamp_s, area_ratio)
        else:
            samples.append((timestamp_s, area_ratio))

        if area_ratio < self.minimum_area_ratio:
            # Far-away boxes are a few pixels wide; their growth is mostly jitter.
            return None
        if len(samples) < self.minimum_samples:
            return None

        first_timestamp, first_area = samples[0]
        span_s = timestamp_s - first_timestamp
        if span_s < self.minimum_span_s:
            return None

        growth_rate = (math.log(area_ratio) - math.log(first_area)) / span_s
        if growth_rate <= 0.0:
            # Shrinking or steady: moving away, or no relative motion at all.
            return None
        consistency = self._growth_consistency(samples)
        if consistency < self.minimum_growth_consistency:
            return None

        time_to_contact_s = 2.0 / growth_rate
        if time_to_contact_s > self.warning_ttc_s:
            return None

        in_path = self._corridor_overlap(left, right, frame_width) >= self.minimum_corridor_overlap
        object_type = normalize_object_type(detection.class_name)
        level = (
            HazardLevel.IMMINENT
            if time_to_contact_s <= self.imminent_ttc_s
            else HazardLevel.WARNING
        )
        if level is HazardLevel.WARNING:
            # A warning is only actionable for something that can cross into the
            # user's path on its own, and only while it is heading down that path.
            if not in_path or object_type not in self.moving_hazard_types:
                return None

        confidence = max(0.0, min(1.0, float(detection.confidence) * consistency))
        if confidence < self.minimum_confidence:
            return None
        if not self._should_emit(history, level, timestamp_s):
            return None

        history.last_emitted_level = level
        history.last_emitted_at_s = timestamp_s
        history.emission_count += 1
        return HazardAssessment(
            stable_id=stable_id,
            object_type=object_type,
            level=level,
            zone=self._zone(left, right, frame_width),
            time_to_contact_s=time_to_contact_s,
            growth_rate=growth_rate,
            area_ratio=area_ratio,
            in_path=in_path,
            confidence=confidence,
            emission_index=history.emission_count,
        )

    @staticmethod
    def _growth_consistency(samples: Sequence[tuple[float, float]]) -> float:
        """Fraction of consecutive steps in which the box actually grew."""
        steps = len(samples) - 1
        if steps <= 0:
            return 0.0
        grew = sum(
            1 for index in range(steps) if samples[index + 1][1] > samples[index][1]
        )
        return grew / steps

    def _corridor_overlap(self, left: float, right: float, frame_width: int) -> float:
        """How much of the box falls inside the user's walking corridor, 0..1."""
        box_width = right - left
        if box_width <= 0.0:
            return 0.0
        half = frame_width * self.corridor_ratio / 2.0
        corridor_left = frame_width / 2.0 - half
        corridor_right = frame_width / 2.0 + half
        overlap = min(right, corridor_right) - max(left, corridor_left)
        return max(0.0, overlap) / box_width

    @staticmethod
    def _zone(left: float, right: float, frame_width: int) -> HazardZone:
        center = (left + right) / 2.0
        third = frame_width / 3.0
        if center < third:
            return HazardZone.LEFT
        if center > 2.0 * third:
            return HazardZone.RIGHT
        return HazardZone.CENTER

    def _should_emit(
        self,
        history: _TrackHistory,
        level: HazardLevel,
        timestamp_s: float,
    ) -> bool:
        previous_level = history.last_emitted_level
        if previous_level is None:
            return True
        if previous_level is HazardLevel.WARNING and level is HazardLevel.IMMINENT:
            # Escalation must not wait out the repeat interval.
            return True
        last_at = history.last_emitted_at_s
        if last_at is None:
            return True
        # While the danger persists the user needs to keep hearing it, but not
        # on every frame.
        return timestamp_s - last_at >= self.repeat_interval_s


def to_analysis_event(assessment: HazardAssessment, timestamp_s: float) -> AnalysisEvent:
    """Wrap one assessment so the narration policy can rank and template it."""
    return AnalysisEvent(
        event_type=HAZARD_DETECTED,
        object_type=assessment.object_type,
        stable_id=assessment.stable_id,
        timestamp_s=timestamp_s,
        current_state=assessment.level.value,
        confidence=assessment.confidence,
        attributes={
            "hazard_level": assessment.level.value,
            "zone": assessment.zone.value,
            "time_to_contact_s": round(assessment.time_to_contact_s, 2),
            "in_path": assessment.in_path,
            "emission_index": assessment.emission_index,
        },
        is_uncertain=False,
    )
