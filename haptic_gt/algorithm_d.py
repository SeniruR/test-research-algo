"""Algorithm D: HapticGen — RMS-driven NCO around 200 Hz."""

from __future__ import annotations

import numpy as np

from .normalize import peak_limit, rms_normalize

FRAME_MS = 10.0
BASE_FREQ_HZ = 200.0
FREQ_DEVIATION_HZ = 50.0


def haptic_gen(
    audio: np.ndarray,
    sr: int,
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    10 ms RMS energy modulates a 200 Hz ± 50 Hz numerically controlled oscillator.
    """
    frame_len = max(1, int(sr * FRAME_MS / 1000))
    hop = frame_len
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop)

    rms_values = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        rms_values[i] = float(np.sqrt(np.mean(frame**2) + 1e-12))

    rms_norm = rms_values / (rms_values.max() + 1e-12)
    # Map normalized energy to frequency deviation around base carrier.
    inst_freq = BASE_FREQ_HZ + (rms_norm * 2.0 - 1.0) * FREQ_DEVIATION_HZ
    inst_freq = np.clip(inst_freq, BASE_FREQ_HZ - FREQ_DEVIATION_HZ, BASE_FREQ_HZ + FREQ_DEVIATION_HZ)

    inst_freq_full = np.interp(
        np.arange(len(audio)),
        np.linspace(0, len(audio) - 1, len(inst_freq)),
        inst_freq,
    ).astype(np.float64)

    phase = np.cumsum(2.0 * np.pi * inst_freq_full / sr)
    synthesized = np.sin(phase).astype(np.float32)

    out = rms_normalize(synthesized, target_rms)
    return peak_limit(out)
