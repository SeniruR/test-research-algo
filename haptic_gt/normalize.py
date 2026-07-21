"""Shared audio normalization utilities."""

from __future__ import annotations

import numpy as np


def rms(signal: np.ndarray) -> float:
    """Root-mean-square amplitude."""
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def rms_normalize(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale signal so its RMS matches target_rms."""
    current = rms(signal)
    if current < 1e-10:
        return signal.astype(np.float32)
    return (signal * (target_rms / current)).astype(np.float32)


def peak_limit(signal: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Soft-limit peaks to avoid clipping on export."""
    max_val = float(np.max(np.abs(signal)))
    if max_val <= peak:
        return signal
    return (signal * (peak / max_val)).astype(np.float32)
