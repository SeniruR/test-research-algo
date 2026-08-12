"""Parse hand-labeled events for HITL ground-truth generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def _as_event_dicts(spec: list[dict[str, Any]] | dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    if isinstance(spec, (str, Path)):
        payload = json.loads(Path(spec).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, dict):
            events = payload.get("events")
            if isinstance(events, list):
                return list(events)
            if "category" in payload and "peak_sec" in payload:
                return [payload]
        raise ValueError(f"Unsupported manual events JSON shape: {spec}")
    if isinstance(spec, dict):
        if "events" in spec and isinstance(spec["events"], list):
            return list(spec["events"])
        return [spec]
    return list(spec)


def events_from_manual(
    spec: list[dict[str, Any]] | dict[str, Any] | str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """
    Build DetectedEvent list from hand labels.

    Each event needs ``category`` and ``peak_sec``. ``start_sec`` / ``end_sec``
    default to a short window around the peak when omitted.
    """
    taxonomy = taxonomy or load_taxonomy()
    rows = _as_event_dicts(spec)
    if not rows:
        return []

    half = taxonomy.impulsive_event_half_width_sec
    pre_roll = taxonomy.impulsive_pre_roll_sec
    events: list[DetectedEvent] = []

    for row in rows:
        category = str(row["category"]).strip()
        if category not in taxonomy.categories:
            raise ValueError(
                f"Unknown category {category!r}. "
                f"Expected one of: {sorted(taxonomy.categories)}"
            )
        peak = float(row["peak_sec"])
        start = float(row["start_sec"]) if row.get("start_sec") is not None else max(0.0, peak - pre_roll)
        end = float(row["end_sec"]) if row.get("end_sec") is not None else peak + half
        if end < start:
            raise ValueError(f"end_sec ({end}) must be >= start_sec ({start})")
        peak = min(max(peak, start), end)
        label = str(row.get("label") or category)
        confidence = float(row.get("confidence", 1.0))
        events.append(
            DetectedEvent(
                category=category,
                label=label,
                start_sec=start,
                peak_sec=peak,
                end_sec=end,
                confidence=confidence,
                context_token=False,
                audio_score=None,
                video_score=None,
                sources=["manual"],
            )
        )

    events.sort(key=lambda e: e.start_sec)
    return events


def vehicle_events_from_peaks(
    peaks_sec: list[float],
    *,
    half_width_sec: float = 0.45,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """
    Build vehicle rumble events from hand-marked peak times (HITL GT).

    Use when auto vehicle detection cannot match ears — calib showed many true
    rumbles have almost no RMS rise, while bed false-positives often do.
    """
    taxonomy = taxonomy or load_taxonomy()
    if "vehicle" not in taxonomy.categories:
        raise ValueError("taxonomy has no vehicle category")
    events: list[DetectedEvent] = []
    for peak in sorted(float(p) for p in peaks_sec):
        start = max(0.0, peak - half_width_sec * 0.35)
        end = peak + half_width_sec
        events.append(
            DetectedEvent(
                category="vehicle",
                label="vehicle_rumble",
                start_sec=start,
                peak_sec=peak,
                end_sec=end,
                confidence=1.0,
                context_token=False,
                audio_score=None,
                video_score=None,
                sources=["manual"],
            )
        )
    return events
