"""Algorithm C: Bark-band pitch tracking and sine synthesis."""

from __future__ import annotations

import numpy as np

from .normalize import peak_limit, rms_normalize

FRAME_MS = 10.0
N_BARK_BANDS = 24
MIN_HAPTIC_HZ = 20.0
MAX_HAPTIC_HZ = 250.0


def _hz_to_bark(freq_hz: np.ndarray) -> np.ndarray:
    f = np.maximum(freq_hz, 1.0)
    return (
        13.0 * np.arctan(0.00076 * f)
        + 3.5 * np.arctan((f / 7500.0) ** 2)
    )


def _bark_to_hz(bark: np.ndarray) -> np.ndarray:
    """Approximate inverse via lookup table interpolation."""
    hz_grid = np.linspace(MIN_HAPTIC_HZ, 8000.0, 8000)
    bark_grid = _hz_to_bark(hz_grid)
    return np.interp(bark, bark_grid, hz_grid)


def _bark_band_edges(n_bands: int = N_BARK_BANDS) -> np.ndarray:
    bark_min, bark_max = 0.0, 24.0
    return np.linspace(bark_min, bark_max, n_bands + 1)


def _dominant_freq_per_frame(frame: np.ndarray, sr: int) -> float:
    """Pick dominant Bark band and return its center frequency."""
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    edges = _bark_band_edges()
    band_energy = np.zeros(N_BARK_BANDS, dtype=np.float64)

    bark_vals = _hz_to_bark(freqs)
    for i in range(N_BARK_BANDS):
        mask = (bark_vals >= edges[i]) & (bark_vals < edges[i + 1])
        if np.any(mask):
            band_energy[i] = np.sum(spectrum[mask] ** 2)

    if band_energy.sum() < 1e-12:
        return 100.0

    peak_idx = int(np.argmax(band_energy))
    center_bark = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    freq = float(_bark_to_hz(np.array([center_bark]))[0])
    return float(np.clip(freq, MIN_HAPTIC_HZ, MAX_HAPTIC_HZ))


def _synthesize_variable_sine(freq_contour: np.ndarray, sr: int) -> np.ndarray:
    """Phase-accumulator sine from instantaneous frequency contour."""
    phase = np.cumsum(2.0 * np.pi * freq_contour / sr)
    return np.sin(phase).astype(np.float32)


def pitch_matching(
    audio: np.ndarray,
    sr: int,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    10 ms windows -> dominant Bark-band frequency -> interpolated sine synthesis.
    """
    frame_len = max(1, int(sr * FRAME_MS / 1000))
    hop = frame_len
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop)

    freqs = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        freqs[i] = _dominant_freq_per_frame(frame, sr)

    # Smooth contour to reduce jitter between 10 ms frames.
    kernel = np.ones(5, dtype=np.float32) / 5.0
    freqs = np.convolve(freqs, kernel, mode="same")

    inst_freq = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio) - 1, len(freqs)),
        freqs,
    ).astype(np.float32)

    synthesized = _synthesize_variable_sine(inst_freq, sr)
    out = rms_normalize(synthesized, target_rms)
    return peak_limit(out)
