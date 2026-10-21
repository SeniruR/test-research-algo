"""Snap impulsive events onto visible fireballs / muzzle flashes.

Audio-only SED follows the soundtrack. Montage clips (this tank cut) put the
boom, the cut, and the orange flash on different frames, so a peak that is
correct in the WAV still looks late or on the wrong shot. This pass is *not*
ViViT: it finds frames where orange/fire pixels suddenly appear.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .frozen_fusion import DetectedEvent
from .taxonomy import Taxonomy, load_taxonomy

_IMPULSIVE = frozenset({"explosion", "gunshot"})


class FireScan(NamedTuple):
    times: np.ndarray
    #: Fraction of orange pixels, and of bright pixels, per frame.
    warm: np.ndarray
    hot: np.ndarray
    #: How well the non-orange background before a frame matches the one after
    #: it (correlation of blurred luma); NaN where too little background is left.
    continuity: np.ndarray


#: Thumbnail size the scan works at, and the blur that hides film grain so that
#: shaky, grainy footage still reads as one shot.
_SCAN_SIZE = (160, 90)
_GRAIN_BLUR = (9, 9)
#: Below this fraction of background (outside the fire) a cut cannot be told apart.
_MIN_BACKGROUND_FRAC = 0.15


def flashes_from_frame_metrics(
    times: np.ndarray,
    warm: np.ndarray,
    hot: np.ndarray,
    *,
    min_d_warm: float,
    min_warm: float,
    min_d_hot: float,
    min_sep_sec: float,
    continuity: np.ndarray | None = None,
    min_continuity: float = 0.0,
    settle_sec: float = 0.0,
) -> list[float]:
    """Return onset times of orange flashes, strongest first then spaced.

    With ``continuity``, an orange jump only counts when the shot carries on
    through it: a cut to a sunset or a red title card brings orange in too.
    Within ``settle_sec`` after a cut, a jump also has to start from a shot that
    opened without orange; otherwise it is the cut's own fire filling in, not a
    new flash (a muzzle flash 0.1 s into a new shot still counts).
    """
    if times.size == 0:
        return []
    d_warm = np.diff(warm, prepend=float(warm[0]))
    d_hot = np.diff(hot, prepend=float(hot[0]))
    candidate = (d_warm >= min_d_warm) & (warm >= min_warm) & (d_hot >= min_d_hot)
    if continuity is not None:
        steady = np.nan_to_num(continuity, nan=-1.0) >= min_continuity
        dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        settle = max(0, int(round(settle_sec / dt))) if dt > 0 else 0
        broken = np.flatnonzero(~steady)
        for i in np.flatnonzero(candidate):
            if not steady[i]:
                candidate[i] = False
                continue
            j = np.searchsorted(broken, i) - 1
            if j >= 0 and broken[j] >= i - settle and warm[broken[j] + 1] >= min_warm:
                candidate[i] = False
    idxs = np.flatnonzero(candidate)
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


def background_continuity(
    before_luma: np.ndarray,
    before_warm: np.ndarray,
    after_luma: np.ndarray,
    after_warm: np.ndarray,
) -> float:
    """Correlation of the blurred luma outside the fire in two frames.

    A fireball lights up and covers part of a shot but leaves the rest of it in
    place (0.97-1.0 measured, grainy handheld film included); a cut replaces it
    (0.2-0.66 on cuts to sunset-lit shots). Correlation ignores the brightness
    change the fire itself causes.
    """
    import cv2

    keep = ~(before_warm | after_warm)
    keep = cv2.erode(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if keep.mean() < _MIN_BACKGROUND_FRAC:
        return float("nan")
    a = cv2.GaussianBlur(before_luma, _GRAIN_BLUR, 3)[keep]
    b = cv2.GaussianBlur(after_luma, _GRAIN_BLUR, 3)[keep]
    if a.std() < 1e-3 or b.std() < 1e-3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def scan_fire_pixels(video_path: str | Path) -> FireScan | None:
    """Orange and bright pixel fractions per frame, and shot continuity."""
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    # Frame i is judged on frame i - lag against frame i + 1, so a jump spread
    # over a couple of frames is still compared across the whole change.
    lag = max(1, int(round(0.1 * fps)))
    times: list[float] = []
    warm: list[float] = []
    hot: list[float] = []
    continuity: list[float] = []
    recent: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=lag + 2)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, _SCAN_SIZE, interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(small)
        luma = 0.114 * b + 0.587 * g + 0.299 * r
        warm_mask = (r > 160) & (r > g + 15) & (r > b + 20)
        times.append(i / fps)
        warm.append(float(warm_mask.mean()))
        hot.append(float((luma > 200).mean()))
        continuity.append(float("nan"))
        recent.append((luma.astype(np.float32), warm_mask))
        if i >= 1:
            before = recent[0]
            continuity[i - 1] = background_continuity(before[0], before[1], *recent[-1])
        i += 1
    cap.release()
    if not times:
        return None
    return FireScan(
        np.asarray(times, dtype=np.float64),
        np.asarray(warm, dtype=np.float64),
        np.asarray(hot, dtype=np.float64),
        np.asarray(continuity, dtype=np.float64),
    )


def flashes_from_scan(scan: FireScan | None, taxonomy: Taxonomy) -> list[float]:
    if scan is None:
        return []
    return flashes_from_frame_metrics(
        scan.times,
        scan.warm,
        scan.hot,
        min_d_warm=taxonomy.visual_flash_min_d_warm,
        min_warm=taxonomy.visual_flash_min_warm,
        min_d_hot=taxonomy.visual_flash_min_d_hot,
        min_sep_sec=taxonomy.visual_flash_min_sep_sec,
        continuity=scan.continuity,
        min_continuity=taxonomy.visual_flash_min_continuity,
        settle_sec=taxonomy.visual_flash_settle_sec,
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
