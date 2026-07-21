"""Object-specific analyzers with a shared structured-result interface."""

from .base import ObjectAnalyzer
from .bus import BusAnalyzer
from .generic import GenericVisionAnalyzer
from .kiosk import KioskAnalyzer
from .text_object import TextObjectAnalyzer
from .traffic_light import TrafficLightAnalyzer

__all__ = [
    "BusAnalyzer",
    "GenericVisionAnalyzer",
    "KioskAnalyzer",
    "ObjectAnalyzer",
    "TextObjectAnalyzer",
    "TrafficLightAnalyzer",
]
