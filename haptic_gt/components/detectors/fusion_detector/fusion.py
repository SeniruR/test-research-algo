"""Join the audio branch and the video branch into one event list.

Rules:
- Audio owns start, peak, and end of every audio event.
- A visible flash near an audio bang moves its peak onto the flash (then the
  peak re-snaps to the audio attack); a flash with no bang adds one accent.
- A short Qwen gunshot/explosion scene may promote a sharp attack the audio
  branch turned down only for lack of backing. A scene never adds an event by
  itself.
- No audio event is dropped for lacking a flash or a scene.
"""

from __future__ import annotations

from pathlib import Path

from .branches import AudioBranchResult, VideoBranchResult
from .frozen_fusion import DetectedEvent, dedupe_events_by_peak
from .impulsive_nms import (
    snap_impulsive_peaks_to_attacks,
    suppress_impulsive_overlaps,
)
from .impulsive_promote import drop_chips_under_blasts
from .onset_refine import refine_event_timing
from .taxonomy import Taxonomy
from .visual_flash import align_impulsive_events_to_flashes
from .visual_scenes import BANG_SCENE_CATEGORIES, VisualSpan

# A candidate this close to an audio bang is that bang, not a new one.
_CANDIDATE_NEIGHBOUR_SEC = 0.40


def _is_impulsive(ev: DetectedEvent, taxonomy: Taxonomy) -> bool:
    cfg = taxonomy.categories.get(ev.category)
    return bool(cfg is not None and cfg.impulsive)


def bang_scene_categories(
    scenes: list[VisualSpan], t: float, *, max_scene_sec: float | None = None
) -> list[str]:
    """Picture bang categories (gunshot / explosion) covering time ``t``.

    A long scene says a fireball is somewhere in it, not that a new bang
    happens at ``t``: the fire stays on screen while the sound decays.
    ``max_scene_sec`` keeps only scenes short enough to pin the attack down.
    """
    found: list[str] = []
    for s in scenes:
        if s.category not in BANG_SCENE_CATEGORIES or not s.start_sec <= t <= s.end_sec:
            continue
        if max_scene_sec is not None and s.end_sec - s.start_sec > max_scene_sec:
            continue
        if s.category not in found:
            found.append(s.category)
    return found


def promote_scene_backed(
    events: list[DetectedEvent],
    candidates: list[DetectedEvent],
    scenes: list[VisualSpan],
    taxonomy: Taxonomy,
) -> tuple[list[DetectedEvent], int]:
    """Add the candidates that sit inside a picture gunshot/explosion scene."""
    if not candidates or not scenes:
        return list(events), 0
    bangs = [e for e in events if _is_impulsive(e, taxonomy)]
    added: list[DetectedEvent] = []
    for cand in sorted(candidates, key=lambda e: e.peak_sec):
        cats = bang_scene_categories(
            scenes, cand.peak_sec, max_scene_sec=taxonomy.visual_scene_backing_max_sec
        )
        if not cats:
            continue
        if any(
            abs(cand.peak_sec - b.peak_sec) <= _CANDIDATE_NEIGHBOUR_SEC
            for b in bangs + added
        ):
            continue
        category = cand.category if cand.category in cats else cats[0]
        cfg = taxonomy.categories.get(category)
        label = cfg.audioset_labels[0] if cfg and cfg.audioset_labels else category
        added.append(
            DetectedEvent(
                category=category,
                label=label,
                start_sec=cand.start_sec,
                peak_sec=cand.peak_sec,
                end_sec=cand.end_sec,
                confidence=cand.confidence,
                context_token=False,
                audio_score=cand.audio_score,
                video_score=None,
                sources=list(cand.sources) + ["visual_scene"],
            )
        )
    if not added:
        return list(events), 0
    kept = added + drop_chips_under_blasts(events, added, taxonomy)
    kept.sort(key=lambda e: e.start_sec)
    return kept, len(added)


def annotate_scene_labels(
    events: list[DetectedEvent],
    scenes: list[VisualSpan],
) -> list[DetectedEvent]:
    """Record which picture scenes each event overlaps. Times are untouched."""
    for ev in events:
        cats: list[str] = []
        for s in scenes:
            if ev.start_sec < s.end_sec and s.start_sec < ev.end_sec and s.category not in cats:
                cats.append(s.category)
        ev.visual_categories = cats
    return events


def fuse_branches(
    audio: AudioBranchResult,
    video: VideoBranchResult,
    source_wav: str | Path,
    taxonomy: Taxonomy,
) -> tuple[list[DetectedEvent], dict]:
    """One event list from both branches, plus a report of what each step changed."""
    source_wav = Path(source_wav)
    events = list(audio.events)

    backing_on = taxonomy.visual_scene_backing and video.qwen_ran
    scene_backed = 0
    if backing_on:
        events, scene_backed = promote_scene_backed(
            events, audio.candidates, video.scenes, taxonomy
        )

    flashes_snapped = flashes_added = 0
    if video.flashes:
        events = align_impulsive_events_to_flashes(
            events, taxonomy=taxonomy, flashes=video.flashes
        )
        for e in events:
            if "visual_flash" not in e.sources:
                continue
            if e.sources == ["visual_flash"]:
                flashes_added += 1
            else:
                flashes_snapped += 1
        # Picture flash can lead the boom by ~0.2 s; snap again onto the sharp attack
        events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
        events = refine_event_timing(
            events, source_wav, taxonomy, relocate_impulsive_peaks=False
        )
        events = dedupe_events_by_peak(events)
        events = suppress_impulsive_overlaps(events, source_wav, taxonomy)
        events = dedupe_events_by_peak(events)

    events = annotate_scene_labels(events, video.scenes)

    report = {
        "audio_events": len(audio.events),
        "audio_unbacked_attacks": len(audio.candidates),
        "flashes_seen": len(video.flashes),
        "flashes_snapped": flashes_snapped,
        "flashes_added": flashes_added,
        "qwen_ran": video.qwen_ran,
        "scene_spans": len(video.scenes),
        "scene_backing": backing_on,
        "scene_backed_added": scene_backed,
        "events_with_scene_label": sum(1 for e in events if e.visual_categories),
        "fused_events": len(events),
    }
    return events, report
