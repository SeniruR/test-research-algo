"""Merge nearby sustained events into longer rumble spans."""

from __future__ import annotations

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def merge_sustained_events(
    events: list[DetectedEvent],
    taxonomy: Taxonomy | None = None,
    *,
    gap_sec: float | None = None,
) -> list[DetectedEvent]:
    """
    Collapse fragmented sustained detections (vehicle chips) into fewer spans.

    Impulsive events are left unchanged. Sustained events of the same category
    whose windows are within ``gap_sec`` are merged; peak/confidence follow the
    strongest member.
    """
    taxonomy = taxonomy or load_taxonomy()
    gap = gap_sec if gap_sec is not None else taxonomy.sustained_merge_gap_sec

    impulsive: list[DetectedEvent] = []
    by_cat: dict[str, list[DetectedEvent]] = {}
    for ev in events:
        cat = taxonomy.categories.get(ev.category)
        if cat is not None and cat.impulsive:
            impulsive.append(ev)
            continue
        by_cat.setdefault(ev.category, []).append(ev)

    merged: list[DetectedEvent] = list(impulsive)
    for cat_name, group in by_cat.items():
        ordered = sorted(group, key=lambda e: e.start_sec)
        if not ordered:
            continue
        cur = ordered[0]
        for ev in ordered[1:]:
            if ev.start_sec <= cur.end_sec + gap:
                best = ev if ev.confidence >= cur.confidence else cur
                cur = DetectedEvent(
                    category=cur.category,
                    label=best.label,
                    start_sec=min(cur.start_sec, ev.start_sec),
                    peak_sec=best.peak_sec,
                    end_sec=max(cur.end_sec, ev.end_sec),
                    confidence=max(cur.confidence, ev.confidence),
                    context_token=cur.context_token or ev.context_token,
                    audio_score=_max_opt(cur.audio_score, ev.audio_score),
                    video_score=_max_opt(cur.video_score, ev.video_score),
                    sources=sorted(set(cur.sources) | set(ev.sources)),
                )
            else:
                merged.append(cur)
                cur = ev
        merged.append(cur)

    merged.sort(key=lambda e: e.start_sec)
    return merged


def _max_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def sustained_coverage_sec(events: list[DetectedEvent], taxonomy: Taxonomy) -> float:
    """Union duration of non-impulsive event windows."""
    intervals = [
        (e.start_sec, e.end_sec)
        for e in events
        if (cat := taxonomy.categories.get(e.category)) is not None and not cat.impulsive
    ]
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = s, e
    total += max(0.0, cur_e - cur_s)
    return total
