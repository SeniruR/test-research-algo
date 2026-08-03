"""Build gated audio from detected event segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.audio_io import INPUT_SR
from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


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


def build_event_mask(
    duration_samples: int,
    sample_rate: int,
    events: list[DetectedEvent],
    *,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Binary mask with short linear fades at segment edges."""
    mask = np.zeros(duration_samples, dtype=np.float32)
    fade = max(1, int(sample_rate * fade_ms / 1000.0))

    for ev in events:
        s0 = max(0, int(ev.start_sec * sample_rate))
        s1 = min(duration_samples, int(ev.end_sec * sample_rate))
        if s1 <= s0:
            continue
        mask[s0:s1] = 1.0
        f0 = min(fade, (s1 - s0) // 2)
        if f0 > 0:
            ramp = np.linspace(0.0, 1.0, f0, dtype=np.float32)
            mask[s0 : s0 + f0] = np.maximum(mask[s0 : s0 + f0], ramp)
            mask[s1 - f0 : s1] = np.maximum(mask[s1 - f0 : s1], ramp[::-1])

    return mask


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
    source_wav = Path(source_wav)
    output_wav = Path(output_wav)
    taxonomy = taxonomy or load_taxonomy()
    gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_categories)

    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    mask = build_event_mask(len(audio), sr, gate_events)
    gated = audio * mask

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, np.clip(gated, -1.0, 1.0), sr, subtype="PCM_16")
    return output_wav
