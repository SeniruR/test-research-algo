"""Events from frame-level posteriors: median filter + hysteresis thresholding.

This is the standard DCASE sound-event-detection decoding step. A category is
"on" once its posterior crosses the high threshold and stays on until it falls
below the low threshold, which stops one noisy frame from chopping an event in
two (or from starting one).
"""

from __future__ import annotations

import numpy as np

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.sed_frames import FramePosteriors
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def median_filter(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0 or k <= 1:
        return values
    k = int(k) | 1
    pad = k // 2
    padded = np.pad(values, pad, mode="edge")
    out = np.empty_like(values)
    for i in range(values.size):
        out[i] = float(np.median(padded[i : i + k]))
    return out


def hysteresis_segments(
    values: np.ndarray,
    high: float,
    low: float,
) -> list[tuple[int, int]]:
    """Frame index spans that cross ``high`` and stay above ``low``."""
    segments: list[tuple[int, int]] = []
    n = values.size
    i = 0
    while i < n:
        if values[i] < high:
            i += 1
            continue
        start = i
        while start > 0 and values[start - 1] >= low:
            start -= 1
        end = i
        while end + 1 < n and values[end + 1] >= low:
            end += 1
        if not segments or start > segments[-1][1]:
            segments.append((start, end))
        else:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end))
        i = end + 1
    return segments


def _thresholds(taxonomy: Taxonomy, category: str) -> tuple[float, float]:
    cfg = taxonomy.categories.get(category)
    high = taxonomy.sed_onset_high
    low = taxonomy.sed_onset_low
    if cfg is not None:
        if cfg.sed_high is not None:
            high = float(cfg.sed_high)
        if cfg.sed_low is not None:
            low = float(cfg.sed_low)
    return high, max(min(low, high), 0.0)


def events_from_frame_posteriors(
    frames: FramePosteriors,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """Decode frame posteriors into events with start / peak / end."""
    taxonomy = taxonomy or load_taxonomy()
    if frames.times.size == 0:
        return []

    hop = max(frames.hop_sec, 1e-4)
    events: list[DetectedEvent] = []

    for category, raw in frames.scores.items():
        cfg = taxonomy.categories.get(category)
        if cfg is None or raw.size == 0:
            continue
        impulsive = bool(cfg.impulsive)
        med_sec = (
            taxonomy.sed_median_impulsive_sec
            if impulsive
            else taxonomy.sed_median_sustained_sec
        )
        smooth = median_filter(raw, max(1, int(round(med_sec / hop))))
        high, low = _thresholds(taxonomy, category)
        min_sec = (
            taxonomy.sed_min_event_sec_impulsive
            if impulsive
            else taxonomy.sed_min_event_sec_sustained
        )

        spans = hysteresis_segments(smooth, high, low)
        merged: list[tuple[int, int]] = []
        gap_frames = max(1, int(round(taxonomy.sed_merge_gap_sec / hop)))
        for s, e in spans:
            if merged and s - merged[-1][1] <= gap_frames:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        label = cfg.audioset_labels[0] if cfg.audioset_labels else category
        for s, e in merged:
            start = float(frames.times[s]) - hop / 2.0
            end = float(frames.times[e]) + hop / 2.0
            if end - start < min_sec:
                pad = (min_sec - (end - start)) / 2.0
                start -= pad
                end += pad
            start = max(0.0, start)
            end = min(frames.duration_sec, max(end, start + min_sec))
            peak_i = s + int(np.argmax(smooth[s : e + 1]))
            events.append(
                DetectedEvent(
                    category=category,
                    label=label,
                    start_sec=start,
                    peak_sec=float(np.clip(frames.times[peak_i], start, end)),
                    end_sec=end,
                    confidence=float(np.max(smooth[s : e + 1])),
                    context_token=False,
                    audio_score=float(np.max(raw[s : e + 1])),
                    # Video is intentionally not fused; see encoders.classify_video_frames
                    video_score=None,
                    sources=["audio", "sed"],
                )
            )

    events.sort(key=lambda ev: ev.start_sec)
    return events


def split_impulsive_by_posterior_peaks(
    events: list[DetectedEvent],
    frames: FramePosteriors,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """Split one long impulsive span into one event per posterior peak.

    A volley of shots can stay above threshold continuously; DCASE decoding
    would report a single event, but each shot needs its own haptic accent.
    """
    taxonomy = taxonomy or load_taxonomy()
    hop = max(frames.hop_sec, 1e-4)
    min_dist_frames = max(1, int(round(taxonomy.impulsive_min_peak_distance_sec / hop)))
    out: list[DetectedEvent] = []

    for ev in events:
        cfg = taxonomy.categories.get(ev.category)
        arr = frames.scores.get(ev.category)
        half = taxonomy.impulsive_event_half_width_sec
        if cfg is None or not cfg.impulsive or arr is None:
            out.append(ev)
            continue
        if ev.end_sec - ev.start_sec <= 2 * half:
            out.append(ev)
            continue

        mask = (frames.times >= ev.start_sec) & (frames.times <= ev.end_sec)
        idxs = np.flatnonzero(mask)
        if idxs.size < 3:
            out.append(ev)
            continue
        window = arr[idxs]
        # A flat above-threshold plateau is not a set of peaks: require a strict
        # rise and a value close to the strongest peak in the span.
        floor = float(np.max(window)) * taxonomy.sed_peak_rel
        local: list[int] = []
        for i in range(1, window.size - 1):
            if window[i] < floor:
                continue
            if window[i] > window[i - 1] and window[i] >= window[i + 1]:
                local.append(i)
        local.sort(key=lambda i: float(window[i]), reverse=True)
        kept: list[int] = []
        for i in local:
            if all(abs(i - j) >= min_dist_frames for j in kept):
                kept.append(i)
        if len(kept) <= 1:
            out.append(ev)
            continue

        for i in sorted(kept):
            peak_t = float(frames.times[idxs[i]])
            out.append(
                DetectedEvent(
                    category=ev.category,
                    label=ev.label,
                    start_sec=max(ev.start_sec, peak_t - taxonomy.impulsive_pre_roll_sec),
                    peak_sec=peak_t,
                    end_sec=min(ev.end_sec, peak_t + half),
                    confidence=float(window[i]),
                    context_token=ev.context_token,
                    audio_score=float(window[i]),
                    video_score=None,
                    sources=list(ev.sources),
                )
            )

    out.sort(key=lambda ev: ev.start_sec)
    return out
