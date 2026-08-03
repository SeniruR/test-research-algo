"""Frozen multimodal context detection for haptic gating."""

from haptic_gt.context.detector import EventResult, detect_events
from haptic_gt.context.frozen_fusion import DetectedEvent

__all__ = ["DetectedEvent", "EventResult", "detect_events"]
