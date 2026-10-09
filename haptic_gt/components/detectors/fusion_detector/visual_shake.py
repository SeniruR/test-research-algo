"""Screen shake from the picture, for the video branch (report only).

Phase correlation between consecutive frames gives the whole picture's shift.
A pan or a tracking shot shifts smoothly; a shake shifts fast and keeps
reversing. Subtracting a moving average of the shift leaves the jitter, and
its short-time RMS is the shake strength per frame, in fractions of the frame
width per frame so it does not depend on resolution.

Nothing here changes events or haptics yet: the segments are written to
visual_context.json so they can be checked against what the clip shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .taxonomy import Taxonomy, load_taxonomy

_ANALYSIS_WIDTH = 320
_CURVE_HZ = 10.0


@dataclass
class ShakeSegment:
    start_sec: float
    end_sec: float
    peak_sec: float
    peak_strength: float


@dataclass
class ShakeResult:
    fps: float = 0.0
    segments: list[ShakeSegment] = field(default_factory=list)
    #: Max shake strength per 1 / _CURVE_HZ seconds, from t = 0.
    curve: list[float] = field(default_factory=list)
    curve_hz: float = _CURVE_HZ
    cuts_sec: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 3),
            "units": "fraction of frame width per frame",
            "segments": [
                {
                    "start_sec": round(s.start_sec, 3),
                    "end_sec": round(s.end_sec, 3),
                    "peak_sec": round(s.peak_sec, 3),
                    "peak_strength": round(s.peak_strength, 5),
                }
                for s in self.segments
            ],
            "cuts_sec": [round(t, 3) for t in self.cuts_sec],
            "curve_hz": self.curve_hz,
            "curve": [round(v, 5) for v in self.curve],
        }


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x.copy()
    pad = n // 2
    padded = np.pad(x, (pad, n - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def shake_strength(
    shifts: np.ndarray,
    cut_mask: np.ndarray,
    fps: float,
    *,
    smooth_sec: float,
    rms_sec: float = 0.2,
) -> np.ndarray:
    """Per-frame shake strength from (n, 2) frame-to-frame shifts.

    Shifts at cuts are meaningless, so they are replaced by the previous shift
    before smoothing and their residual is zeroed.
    """
    v = shifts.astype(np.float64).copy()
    for i in np.flatnonzero(cut_mask):
        v[i] = v[i - 1] if i > 0 else 0.0
    n_smooth = max(1, int(round(smooth_sec * fps)))
    smooth = np.stack([_moving_average(v[:, k], n_smooth) for k in range(2)], axis=1)
    residual = np.linalg.norm(v - smooth, axis=1)
    residual[cut_mask] = 0.0
    n_rms = max(1, int(round(rms_sec * fps)))
    return np.sqrt(_moving_average(residual**2, n_rms))


def segments_from_strength(
    strength: np.ndarray,
    fps: float,
    *,
    min_strength: float,
    min_sec: float,
    merge_gap_sec: float,
) -> list[ShakeSegment]:
    """Stretches above ``min_strength``, small gaps bridged, short blips dropped."""
    above = strength >= min_strength
    runs: list[list[int]] = []
    i = 0
    while i < above.size:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < above.size and above[j + 1]:
            j += 1
        if runs and (i - runs[-1][1] - 1) / fps <= merge_gap_sec:
            runs[-1][1] = j
        else:
            runs.append([i, j])
        i = j + 1
    out: list[ShakeSegment] = []
    for a, b in runs:
        if (b - a + 1) / fps < min_sec:
            continue
        k = a + int(np.argmax(strength[a : b + 1]))
        out.append(
            ShakeSegment(
                start_sec=a / fps,
                end_sec=(b + 1) / fps,
                peak_sec=k / fps,
                peak_strength=float(strength[k]),
            )
        )
    return out


def detect_screen_shake(
    video_path: str | Path,
    taxonomy: Taxonomy | None = None,
) -> ShakeResult:
    """Scan every frame of ``video_path`` for camera shake."""
    taxonomy = taxonomy or load_taxonomy()
    try:
        import cv2
    except ImportError:
        return ShakeResult()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ShakeResult()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    cuts: list[bool] = [False]
    prev = None
    window = None
    width = _ANALYSIS_WIDTH
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        size = (width, max(2, int(round(h * width / max(w, 1)))))
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        if window is None:
            window = cv2.createHanningWindow(size, cv2.CV_32F)
        if prev is not None:
            (dx, dy), response = cv2.phaseCorrelate(prev, gray, window)
            # A cut decorrelates the two frames; its "shift" is noise
            is_cut = response < taxonomy.visual_shake_cut_response
            shifts.append((dx / width, dy / width))
            cuts.append(bool(is_cut))
        prev = gray
    cap.release()
    if len(shifts) < 3:
        return ShakeResult(fps=fps)

    shift_arr = np.asarray(shifts, dtype=np.float64)
    cut_mask = np.asarray(cuts, dtype=bool)
    strength = shake_strength(
        shift_arr, cut_mask, fps, smooth_sec=taxonomy.visual_shake_smooth_sec
    )
    segments = segments_from_strength(
        strength,
        fps,
        min_strength=taxonomy.visual_shake_min_strength,
        min_sec=taxonomy.visual_shake_min_sec,
        merge_gap_sec=taxonomy.visual_shake_merge_gap_sec,
    )
    per_bin = max(1, int(round(fps / _CURVE_HZ)))
    curve = [
        float(strength[i : i + per_bin].max()) for i in range(0, strength.size, per_bin)
    ]
    return ShakeResult(
        fps=fps,
        segments=segments,
        curve=curve,
        cuts_sec=[i / fps for i in np.flatnonzero(cut_mask)],
    )
