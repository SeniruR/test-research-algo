"""Keep only event spans of the source audio."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from haptic_gt.core.audio_io import INPUT_SR


class _Span(Protocol):
    start_sec: float
    end_sec: float


def build_event_mask(
    duration_samples: int,
    sample_rate: int,
    events: Iterable[_Span],
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
    events: Iterable[_Span],
    *,
    sample_rate: int = INPUT_SR,
) -> Path:
    """Write a mono WAV that keeps the audio under ``events`` and silences the rest."""
    source_wav = Path(source_wav)
    output_wav = Path(output_wav)

    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    mask = build_event_mask(len(audio), sr, list(events))
    gated = audio * mask

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, np.clip(gated, -1.0, 1.0), sr, subtype="PCM_16")
    return output_wav
