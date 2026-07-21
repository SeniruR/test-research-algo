"""Algorithm B: Frequency shifting into the haptic band."""

from __future__ import annotations

import numpy as np
import librosa
from scipy import signal

from .normalize import peak_limit, rms_normalize


def _bandpass_haptic(audio: np.ndarray, sr: int) -> np.ndarray:
    """Keep energy in roughly 10–250 Hz (skin-sensitive range)."""
    sos = signal.butter(4, [10, 250], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _octave_down(audio: np.ndarray, sr: int, octaves: int) -> np.ndarray:
    """
    Fast octave-down via resampling (-12 semitones per octave).
    Much faster than librosa pitch_shift for long clips (3–5 min).
    """
    factor = 0.5**octaves
    down_sr = max(1, int(sr * factor))
    down = librosa.resample(audio, orig_sr=sr, target_sr=down_sr)
    up = librosa.resample(down, orig_sr=down_sr, target_sr=sr)
    if len(up) < len(audio):
        up = np.pad(up, (0, len(audio) - len(up)))
    return up[: len(audio)].astype(np.float32)


def frequency_shifting(
    audio: np.ndarray,
    sr: int,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Down-shift by -12 and -24 semitones, combine with band-limited original,
    bandpass 10–250 Hz, then RMS-normalize.
    """
    shifted_12 = _octave_down(audio, sr, octaves=1)
    shifted_24 = _octave_down(audio, sr, octaves=2)

    original_band = _bandpass_haptic(audio, sr)
    s12_band = _bandpass_haptic(shifted_12, sr)
    s24_band = _bandpass_haptic(shifted_24, sr)

    combined = 0.35 * original_band + 0.35 * s12_band + 0.30 * s24_band
    filtered = _bandpass_haptic(combined, sr)

    out = rms_normalize(filtered, target_rms)
    return peak_limit(out)
