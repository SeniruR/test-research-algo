"""Snap impulsive events onto visible fireballs / muzzle flashes.

Audio-only SED follows the soundtrack. Montage clips (this tank cut) put the
boom, the cut, and the orange flash on different frames, so a peak that is
correct in the WAV still looks late or on the wrong shot. This pass is *not*
ViViT: it finds frames where orange/fire pixels suddenly appear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .frozen_fusion import DetectedEvent
from .taxonomy import Taxonomy, load_taxonomy

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


def scan_fire_pixels(
    video_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-frame times, orange-pixel fraction and bright-pixel fraction."""
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
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
        return None
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(warm, dtype=np.float64),
        np.asarray(hot, dtype=np.float64),
    )


def flashes_from_scan(
    scan: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    taxonomy: Taxonomy,
) -> list[float]:
    if scan is None:
        return []
    times, warm, hot = scan
    return flashes_from_frame_metrics(
        times,
        warm,
        hot,
        min_d_warm=taxonomy.visual_flash_min_d_warm,
        min_warm=taxonomy.visual_flash_min_warm,
        min_d_hot=taxonomy.visual_flash_min_d_hot,
        min_sep_sec=taxonomy.visual_flash_min_sep_sec,
    )


def detect_visual_flashes(
    video_path: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[float]:
    """Scan ``video_path`` for orange-pixel onsets."""
    taxonomy = taxonomy or load_taxonomy()
    return flashes_from_scan(scan_fire_pixels(video_path), taxonomy)


def fireball_spans(
    times: np.ndarray,
    warm: np.ndarray,
    flashes: list[float],
    *,
    min_warm: float,
    max_sec: float,
    gap_sec: float,
) -> list[tuple[float, float]]:
    """How long each flash's fire stays on screen without leaving the frame.

    A span runs from the flash while the orange fraction stays at ``min_warm``,
    tolerating dips up to ``gap_sec``, and is capped at ``max_sec``.
    """
    spans: list[tuple[float, float]] = []
    if times.size == 0:
        return spans
    for flash_t in flashes:
        i = int(np.searchsorted(times, flash_t - 1e-6))
        end_t = flash_t
        last_warm_t = flash_t
        while i < times.size and times[i] <= flash_t + max_sec:
            if warm[i] >= min_warm:
                last_warm_t = float(times[i])
                end_t = last_warm_t
            elif times[i] - last_warm_t > gap_sec:
                break
            i += 1
        spans.append((float(flash_t), end_t))
    return spans


def drop_decay_inside_fireballs(
    events: list[DetectedEvent],
    spans: list[tuple[float, float]],
    taxonomy: Taxonomy,
) -> tuple[list[DetectedEvent], list[float]]:
    """Drop weaker flash-less bangs that sit inside an on-screen fireball.

    The fire that grows on screen after a blast is that blast; the soundtrack
    keeps throwing up attacks while it decays, and on a clipped track they look
    as sharp as real shots. Only explosion-like categories are dropped, only
    without a flash of their own, and only when their attack is weaker than the
    flash-backed blast that started the fire. Returns the kept events and the
    dropped peak times.
    """
    if not spans:
        return list(events), []
    decay_cats = set(taxonomy.visual_fireball_decay_categories)
    match = taxonomy.visual_flash_match_sec
    min_dist = taxonomy.impulsive_min_peak_distance_sec

    def strength(ev: DetectedEvent) -> float:
        return ev.attack_prominence if ev.attack_prominence is not None else 0.0

    dropped: set[int] = set()
    for span_start, span_end in spans:
        parents = [
            e for e in events
            if e.category in _IMPULSIVE
            and "visual_flash" in e.sources
            and abs(e.peak_sec - span_start) <= match
        ]
        if not parents:
            continue
        parent = max(parents, key=strength)
        for i, ev in enumerate(events):
            if (
                ev is parent
                or ev.category not in decay_cats
                or "visual_flash" in ev.sources
                or not parent.peak_sec + min_dist <= ev.peak_sec <= span_end
                or strength(ev) >= strength(parent)
            ):
                continue
            dropped.add(i)
    kept = [e for i, e in enumerate(events) if i not in dropped]
    return kept, sorted({round(events[i].peak_sec, 3) for i in dropped})


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
