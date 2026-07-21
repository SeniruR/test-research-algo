"""Algorithm A: Perception mapping from loudness and roughness to dual sine haptics."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .normalize import peak_limit, rms_normalize

FRAME_MS = 10.0
INTENSITY_CARRIER_HZ = 100.0
ROUGHNESS_CARRIER_HZ = 180.0


def _frame_signal(values: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(values) < frame_len:
        pad = frame_len - len(values)
        values = np.pad(values, (0, pad))
    n_frames = 1 + max(0, (len(values) - frame_len) // hop)
    return np.stack(
        [values[i * hop : i * hop + frame_len] for i in range(n_frames)],
        axis=0,
    )


def _interp_to_length(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(values) == 1:
        return np.full(length, values[0], dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _loudness_contour(audio: np.ndarray, sr: int) -> np.ndarray:
    """Short-time loudness proxy in phon-like units (0–1)."""
    frame_len = int(sr * FRAME_MS / 1000)
    hop = frame_len
    frames = _frame_signal(audio, frame_len, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    db = np.clip(db, db.min(), db.max())
    norm = (db - db.min()) / (db.max() - db.min() + 1e-12)
    return norm.astype(np.float32)


def _roughness_contour(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Roughness proxy from amplitude modulation in the haptic band (15–300 Hz).
    Higher modulation depth implies higher perceived roughness.
    """
    frame_len = int(sr * FRAME_MS / 1000)
    hop = frame_len
    sos = signal.butter(4, [15, 300], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, audio)
    frames = _frame_signal(band, frame_len, hop)
    envelope = np.abs(signal.hilbert(frames, axis=1))
    mod_depth = np.std(envelope, axis=1) / (np.mean(envelope, axis=1) + 1e-12)
    mod_depth = np.clip(mod_depth, 0.0, None)
    norm = mod_depth / (mod_depth.max() + 1e-12)
    return norm.astype(np.float32)


def perception_mapping(
    audio: np.ndarray,
    sr: int,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Map loudness -> intensity carrier and roughness -> roughness carrier.
    Synthesize dual sine waves and RMS-normalize.
    """
    length = len(audio)
    loudness = _interp_to_length(_loudness_contour(audio, sr), length)
    roughness = _interp_to_length(_roughness_contour(audio, sr), length)

    t = np.arange(length, dtype=np.float64) / sr
    intensity = loudness * np.sin(2 * np.pi * INTENSITY_CARRIER_HZ * t)
    rough = roughness * np.sin(2 * np.pi * ROUGHNESS_CARRIER_HZ * t)
    combined = 0.65 * intensity + 0.35 * rough

    out = rms_normalize(combined.astype(np.float32), target_rms)
    return peak_limit(out)
