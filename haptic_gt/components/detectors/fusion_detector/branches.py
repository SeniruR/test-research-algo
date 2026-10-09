"""The two independent detection branches.

The audio branch reads only the soundtrack and the video branch reads only the
picture; neither sees the other's output. ``fusion.fuse_branches`` joins them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .encoders import EncoderScore
from .frozen_fusion import DetectedEvent, dedupe_events_by_peak
from .impulsive_nms import (
    snap_impulsive_peaks_to_attacks,
    suppress_impulsive_overlaps,
)
from .impulsive_promote import promote_impulsive_transients
from .onset_refine import refine_event_timing
from .rumble_filter import filter_sustained_rumble_bursts
from .sed_events import (
    events_from_frame_posteriors,
    split_impulsive_by_posterior_peaks,
)
from .sed_frames import compute_frame_posteriors, posteriors_to_encoder_scores
from .sustained_merge import merge_sustained_events
from .taxonomy import Taxonomy
from .tokenization import _load_audio_16k
from .visual_flash import fireball_spans, flashes_from_scan, scan_fire_pixels
from .visual_scenes import VisualSpan

VISUAL_CONTEXT_NAME = "visual_context.json"


@dataclass
class AudioBranchResult:
    events: list[DetectedEvent]
    #: Sharp attacks turned down only because nothing backed them.
    candidates: list[DetectedEvent]
    encoder_scores: list[EncoderScore]
    gate_report: dict
    detector_info: dict


@dataclass
class VideoBranchResult:
    flashes: list[float] = field(default_factory=list)
    #: (flash, last frame the fire is still on screen) per flash.
    fireballs: list[tuple[float, float]] = field(default_factory=list)
    scenes: list[VisualSpan] = field(default_factory=list)
    qwen_ran: bool = False
    scene_report: dict = field(default_factory=dict)


def run_audio_branch(source_wav: str | Path, taxonomy: Taxonomy) -> AudioBranchResult:
    """Frame SED and flux timing on the soundtrack only (the video is never opened)."""
    source_wav = Path(source_wav)
    audio_16k = _load_audio_16k(source_wav)
    frames = compute_frame_posteriors(audio_16k, taxonomy)
    encoder_scores = posteriors_to_encoder_scores(frames, taxonomy)

    events = events_from_frame_posteriors(frames, taxonomy)
    events = split_impulsive_by_posterior_peaks(events, frames, taxonomy)
    events = refine_event_timing(events, source_wav, taxonomy)
    events = dedupe_events_by_peak(events)
    events, candidates = promote_impulsive_transients(
        events, source_wav, encoder_scores, taxonomy, return_candidates=True
    )
    # SED/refine often sit on a decay bump; snap back to the muzzle in-window
    events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
    events = refine_event_timing(
        events, source_wav, taxonomy, relocate_impulsive_peaks=False
    )
    events = dedupe_events_by_peak(events)
    # One accent per blast: a shot plus a bump in its own decay is one bang
    events = suppress_impulsive_overlaps(events, source_wav, taxonomy)
    events = merge_sustained_events(events, taxonomy)
    gate_report: dict = {}
    events = filter_sustained_rumble_bursts(
        events, source_wav, taxonomy, report=gate_report
    )
    events = dedupe_events_by_peak(events)

    return AudioBranchResult(
        events=events,
        candidates=candidates,
        encoder_scores=encoder_scores,
        gate_report=gate_report,
        detector_info={
            "mode": "frame_sed",
            "backend": frames.backend,
            "frame_hop_sec": round(frames.hop_sec, 4),
            "window_sec": taxonomy.sed_window_sec,
            "onset_high": taxonomy.sed_onset_high,
            "onset_low": taxonomy.sed_onset_low,
            "use_video": taxonomy.use_video,
        },
    )


def run_video_branch(
    video_path: str | Path,
    taxonomy: Taxonomy,
    *,
    use_qwen: bool,
) -> VideoBranchResult:
    """Orange-flash times (every frame) and Qwen scene labels (per scene)."""
    video_path = Path(video_path)
    result = VideoBranchResult()
    if taxonomy.visual_flash_enabled:
        scan = scan_fire_pixels(video_path)
        result.flashes = flashes_from_scan(scan, taxonomy)
        if scan is not None:
            result.fireballs = fireball_spans(
                scan.times,
                scan.warm,
                result.flashes,
                min_warm=taxonomy.visual_flash_min_warm,
                max_sec=taxonomy.visual_fireball_max_sec,
                gap_sec=taxonomy.visual_fireball_gap_sec,
            )
    if use_qwen:
        from .visual_scenes import classify_scenes

        result.scenes, result.scene_report = classify_scenes(video_path, taxonomy)
        result.qwen_ran = True
    return result


def write_visual_context(
    video: VideoBranchResult,
    output_dir: str | Path,
    *,
    video_name: str,
) -> Path:
    """Save what the video branch saw, so a run can be checked without re-running Qwen."""
    out = Path(output_dir) / VISUAL_CONTEXT_NAME
    payload = {
        "video": video_name,
        "note": (
            "Video branch output. Scene spans are picture context, not bang times; "
            "flash times are frame onsets of an orange flash. Audio owns start, peak, and end."
        ),
        "qwen_ran": video.qwen_ran,
        "flashes_sec": [round(float(t), 3) for t in video.flashes],
        "fireballs_sec": [[round(s, 3), round(e, 3)] for s, e in video.fireballs],
        "spans": [
            {
                "category": s.category,
                "start_sec": round(s.start_sec, 3),
                "end_sec": round(s.end_sec, 3),
                "source": "qwen_vl",
            }
            for s in video.scenes
        ],
        "scene_report": video.scene_report,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
