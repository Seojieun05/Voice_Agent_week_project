from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final


class AnalyzerKind(str, Enum):
    """Object-specific analyzer selected for one normalized detector class."""

    TRAFFIC_LIGHT = "traffic_light"
    BUS = "bus"
    KIOSK = "kiosk"
    TEXT = "text"
    GENERIC = "generic"


@dataclass(frozen=True, slots=True)
class ObjectClassSpec:
    """Runtime contract for a detector class emitted by a custom checkpoint."""

    analyzer: AnalyzerKind
    signal_type: str | None = None


def normalize_object_type(class_name: str) -> str:
    """Normalize detector labels without changing the original Detection value."""
    normalized = re.sub(r"[\s-]+", "_", class_name.strip().lower())
    return normalized or "unknown"


_GENERIC_SPEC: Final = ObjectClassSpec(AnalyzerKind.GENERIC)
_SPECS: Final[dict[str, ObjectClassSpec]] = {
    "traffic_light": ObjectClassSpec(AnalyzerKind.TRAFFIC_LIGHT),
    "pedestrian_signal": ObjectClassSpec(
        AnalyzerKind.TRAFFIC_LIGHT,
        signal_type="PEDESTRIAN",
    ),
    "vehicle_traffic_light": ObjectClassSpec(
        AnalyzerKind.TRAFFIC_LIGHT,
        signal_type="VEHICLE",
    ),
    "bus": ObjectClassSpec(AnalyzerKind.BUS),
    "kiosk": ObjectClassSpec(AnalyzerKind.KIOSK),
    "self_service_kiosk": ObjectClassSpec(AnalyzerKind.KIOSK),
    "touchscreen_kiosk": ObjectClassSpec(AnalyzerKind.KIOSK),
    "sign": ObjectClassSpec(AnalyzerKind.TEXT),
    "stop_sign": ObjectClassSpec(AnalyzerKind.TEXT),
    "display": ObjectClassSpec(AnalyzerKind.TEXT),
    "screen": ObjectClassSpec(AnalyzerKind.TEXT),
    "monitor": ObjectClassSpec(AnalyzerKind.TEXT),
    "tv": ObjectClassSpec(AnalyzerKind.TEXT),
    "ticket_machine": ObjectClassSpec(AnalyzerKind.TEXT),
    "bus_route_display": ObjectClassSpec(AnalyzerKind.TEXT),
    "reverse_vending_machine": _GENERIC_SPEC,
    "unknown": _GENERIC_SPEC,
    "unknown_object": _GENERIC_SPEC,
    "unknown_panel": _GENERIC_SPEC,
}

OBJECT_CLASS_SPECS: Final = MappingProxyType(_SPECS)
SIGNAL_OBJECT_TYPES: Final = frozenset(
    name for name, spec in _SPECS.items() if spec.analyzer is AnalyzerKind.TRAFFIC_LIGHT
)
KIOSK_OBJECT_TYPES: Final = frozenset(
    name for name, spec in _SPECS.items() if spec.analyzer is AnalyzerKind.KIOSK
)
TEXT_OBJECT_TYPES: Final = frozenset(
    name for name, spec in _SPECS.items() if spec.analyzer is AnalyzerKind.TEXT
)


def object_class_spec(class_name: str) -> ObjectClassSpec:
    """Return the explicit class contract or the conservative Generic fallback."""
    return OBJECT_CLASS_SPECS.get(normalize_object_type(class_name), _GENERIC_SPEC)


def is_signal_object_type(class_name: str) -> bool:
    return object_class_spec(class_name).analyzer is AnalyzerKind.TRAFFIC_LIGHT


def is_kiosk_object_type(class_name: str) -> bool:
    return object_class_spec(class_name).analyzer is AnalyzerKind.KIOSK
