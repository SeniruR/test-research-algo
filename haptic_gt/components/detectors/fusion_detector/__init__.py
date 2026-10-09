"""Frozen multimodal context detection for haptic gating."""

from .detector import EventResult, detect_events
from .frozen_fusion import DetectedEvent
from .manual_events import events_from_manual

__all__ = ["DetectedEvent", "EventResult", "detect_events", "events_from_manual"]
