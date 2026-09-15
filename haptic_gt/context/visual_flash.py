"""Snap impulsive events onto visible fireballs / muzzle flashes.

Audio-only SED follows the soundtrack. Montage clips (this tank cut) put the
boom, the cut, and the orange flash on different frames, so a peak that is
correct in the WAV still looks late or on the wrong shot. This pass is *not*
ViViT: it finds frames where orange/fire pixels suddenly appear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy

_IMPULSIVE = frozenset({"explosion", "gunshot"})


def flashes_from_frame_metrics(
    times: np.ndarray,
    warm: np.ndarray,
    hot: np.ndarray,
    *,
    min_d_warm: float,
    min_warm: float,
    min_d_hot: float,
    min_sep_sec: float,
) -> list[float]:
    """Return onset times of orange flashes, strongest first then spaced."""
    if times.size == 0:
        return []
    d_warm = np.diff(warm, prepend=float(warm[0]))
    d_hot = np.diff(hot, prepend=float(hot[0]))
    idxs = np.flatnonzero(
        (d_warm >= min_d_warm) & (warm >= min_warm) & (d_hot >= min_d_hot)
    )
    ranked = sorted(
        ((float(d_warm[i]), float(times[i])) for i in idxs),
        reverse=True,
    )
    kept: list[float] = []
    for _, t in ranked:
        if any(abs(t - k) < min_sep_sec for k in kept):
            continue
        kept.append(t)
    kept.sort()
    return kept


def detect_visual_flashes(
    video_path: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[float]:
    """Scan ``video_path`` for orange-pixel onsets."""
    taxonomy = taxonomy or load_taxonomy()
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    times: list[float] = []
    warm: list[float] = []
    hot: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(small)
        luma = 0.114 * b + 0.587 * g + 0.299 * r
        times.append(i / fps)
        warm.append(
            float(((r > 160) & (r > g + 15) & (r > b + 20)).mean())
        )
        hot.append(float((luma > 200).mean()))
        i += 1
    cap.release()
    if not times:
        return []
    return flashes_from_frame_metrics(
        np.asarray(times, dtype=np.float64),
        np.asarray(warm, dtype=np.float64),
        np.asarray(hot, dtype=np.float64),
        min_d_warm=taxonomy.visual_flash_min_d_warm,
        min_warm=taxonomy.visual_flash_min_warm,
        min_d_hot=taxonomy.visual_flash_min_d_hot,
        min_sep_sec=taxonomy.visual_flash_min_sep_sec,
    )


def align_impulsive_events_to_flashes(
    events: list[DetectedEvent],
    video_path: str | Path | None = None,
    taxonomy: Taxonomy | None = None,
    *,
    flashes: list[float] | None = None,
) -> list[DetectedEvent]:
    """Snap / add explosion peaks onto visible flashes; never drop audio fires.

    Matched peaks move onto the fireball frame. Unmatched flashes become new
    accents. Audio/SED bangs with no nearby flash are kept as-is (off-screen
    or low-orange booms still need a haptic hit). Vehicle/weather spans are
    left alone.
    """
    taxonomy = taxonomy or load_taxonomy()
    if not taxonomy.visual_flash_enabled:
        return list(events)

    if flashes is None:
        if video_path is None:
            return list(events)
        flashes = detect_visual_flashes(video_path, taxonomy)
    if not flashes:
        return list(events)

    radius = taxonomy.visual_flash_match_sec
    pre = taxonomy.impulsive_pre_roll_sec
    half = taxonomy.impulsive_event_half_width_sec

    impulsive = [e for e in events if e.category in _IMPULSIVE]
    other = [e for e in events if e.category not in _IMPULSIVE]
    used: set[int] = set()
    out: list[DetectedEvent] = list(other)

    for flash_t in flashes:
        best_i = None
        best_dist = radius
        for i, ev in enumerate(impulsive):
            if i in used:
                continue
            dist = abs(ev.peak_sec - flash_t)
            if dist <= best_dist:
                best_dist = dist
                best_i = i
        if best_i is None:
            out.append(
                DetectedEvent(
                    category="explosion",
                    label="Explosion",
                    start_sec=max(0.0, flash_t - pre),
                    peak_sec=flash_t,
                    end_sec=flash_t + half,
                    confidence=0.55,
                    sources=["visual_flash"],
                    video_score=1.0,
                )
            )
            continue
        used.add(best_i)
        ev = impulsive[best_i]
        sources = list(ev.sources)
        if "visual_flash" not in sources:
            sources.append("visual_flash")
        start = max(0.0, flash_t - pre)
        end = max(ev.end_sec, flash_t + half)
        out.append(
            DetectedEvent(
                category=ev.category,
                label=ev.label,
                start_sec=start,
                peak_sec=flash_t,
                end_sec=max(end, start + 0.2),
                confidence=ev.confidence,
                context_token=ev.context_token,
                audio_score=ev.audio_score,
                video_score=ev.video_score,
                sources=sources,
                attack_rel_max=ev.attack_rel_max,
                attack_prominence=ev.attack_prominence,
            )
        )

    # Keep bangs the picture missed (no orange jump / off-screen boom).
    for i, ev in enumerate(impulsive):
        if i not in used:
            out.append(ev)

    out.sort(key=lambda e: e.start_sec)
    return out
