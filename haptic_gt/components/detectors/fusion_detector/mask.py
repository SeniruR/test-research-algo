"""Pick gate-eligible events from the taxonomy and build gated audio."""

from __future__ import annotations

from pathlib import Path

from haptic_gt.core import gating
from haptic_gt.core.audio_io import INPUT_SR
from haptic_gt.core.gating import build_event_mask

from .frozen_fusion import DetectedEvent
from .taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "apply_gate",
    "build_event_mask",
    "event_included_in_gate",
    "events_for_haptic_gate",
    "resolve_gate_categories",
]


def resolve_gate_categories(
    taxonomy: Taxonomy,
    gate_categories: list[str] | None = None,
) -> list[str]:
    """Return categories used for haptic gating."""
    if gate_categories is not None:
        return list(gate_categories)
    return [
        name
        for name, cat in taxonomy.categories.items()
        if cat.include_in_haptic_gate
    ]


def events_for_haptic_gate(
    events: list[DetectedEvent],
    taxonomy: Taxonomy,
    gate_categories: list[str] | None = None,
) -> list[DetectedEvent]:
    """Events that should contribute to gated_audio.wav."""
    allowed = set(resolve_gate_categories(taxonomy, gate_categories))
    gated: list[DetectedEvent] = []
    for ev in events:
        if ev.category in allowed:
            gated.append(ev)
    return gated


def event_included_in_gate(
    event: DetectedEvent,
    taxonomy: Taxonomy,
    gate_categories: list[str] | None = None,
) -> bool:
    allowed = set(resolve_gate_categories(taxonomy, gate_categories))
    return event.category in allowed


def apply_gate(
    source_wav: str | Path,
    output_wav: str | Path,
    events: list[DetectedEvent],
    *,
    sample_rate: int = INPUT_SR,
    taxonomy: Taxonomy | None = None,
    gate_categories: list[str] | None = None,
) -> Path:
    """Write gated mono WAV keeping only selected event segments."""
    taxonomy = taxonomy or load_taxonomy()
    gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_categories)
    return gating.apply_gate(source_wav, output_wav, gate_events, sample_rate=sample_rate)
