"""Frozen multimodal context detection for haptic gating."""

from haptic_gt.context.detector import EventResult, detect_events
from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.manual_events import events_from_manual

__all__ = ["DetectedEvent", "EventResult", "detect_events", "events_from_manual"]
