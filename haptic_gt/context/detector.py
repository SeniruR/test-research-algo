"""Orchestrate Phase 1 + Phase 2 context detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from haptic_gt.context.context_detectors import symbolic_tokens_from_scores
from haptic_gt.context.encoders import run_encoder_pass
from haptic_gt.context.frozen_fusion import DetectedEvent, dedupe_events_by_peak, fuse_events
from haptic_gt.context.mask import (
    apply_gate,
    event_included_in_gate,
    events_for_haptic_gate,
    resolve_gate_categories,
)
from haptic_gt.context.onset_refine import refine_event_timing
from haptic_gt.context.proposals import propose_all_windows
from haptic_gt.context.impulsive_nms import (
    measure_impulsive_attacks,
    snap_impulsive_peaks_to_attacks,
    suppress_impulsive_overlaps,
)
from haptic_gt.context.impulsive_promote import promote_impulsive_transients
from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
from haptic_gt.context.sed_events import (
    events_from_frame_posteriors,
    split_impulsive_by_posterior_peaks,
)
from haptic_gt.context.sed_frames import (
    compute_frame_posteriors,
    posteriors_to_encoder_scores,
)
from haptic_gt.context.sustained_merge import merge_sustained_events
from haptic_gt.context.taxonomy import load_taxonomy
from haptic_gt.context.tokenization import tokenize_video_audio
from haptic_gt.context.visual_flash import align_impulsive_events_to_flashes

EVENTS_JSON_NAME = "events.json"
GATED_AUDIO_NAME = "gated_audio.wav"


@dataclass
class EventResult:
    events: list[DetectedEvent] = field(default_factory=list)
    no_events_detected: bool = True
    no_haptic_events: bool = True
    timeline_hz: int = 100
    gate_categories_used: list[str] = field(default_factory=list)
    gated_wav: Path | None = None
    events_json: Path | None = None
    haptic_outputs: dict[str, str] = field(default_factory=dict)
    detector_info: dict = field(default_factory=dict)
    sustained_gate: dict = field(default_factory=dict)

    def to_dict(self, *, output_dir: Path | None = None) -> dict:
        taxonomy = load_taxonomy()
        gate_cats = self.gate_categories_used or resolve_gate_categories(taxonomy)

        def _rel(path: str | None) -> str | None:
            if path is None or output_dir is None:
                return path
            try:
                return str(Path(path).relative_to(output_dir))
            except ValueError:
                return path

        event_rows = []
        for i, e in enumerate(self.events, start=1):
            row = {
                "event_id": f"event_{i:03d}",
                "category": e.category,
                "label": e.label,
                "start_sec": round(e.start_sec, 3),
                "peak_sec": round(e.peak_sec, 3),
                "end_sec": round(e.end_sec, 3),
                "confidence": round(e.confidence, 4),
                "context_token": e.context_token,
                "audio_score": e.audio_score,
                "video_score": e.video_score,
                "sources": e.sources,
                "included_in_gate": event_included_in_gate(
                    e, taxonomy, gate_categories=gate_cats
                ),
            }
            # Frame posteriors report the decode threshold for every promoted
            # shot, so strength has to be read off the attack instead.
            if e.attack_rel_max is not None:
                row["attack_rel_max"] = e.attack_rel_max
            if e.attack_prominence is not None:
                row["attack_prominence"] = e.attack_prominence
            event_rows.append(row)

        haptic_out = {
            key: _rel(val) for key, val in self.haptic_outputs.items() if val
        }

        return {
            "detector": self.detector_info,
            "sustained_gate": self.sustained_gate,
            "no_events_detected": self.no_events_detected,
            "no_haptic_events": self.no_haptic_events,
            "gate_categories_used": gate_cats,
            "timeline_hz": self.timeline_hz,
            "haptic_outputs": haptic_out,
            "events": event_rows,
        }


def detect_events(
    video_path: str | Path,
    source_wav: str | Path,
    output_dir: str | Path | None = None,
    *,
    taxonomy_path: str | Path | None = None,
    window_sec: float | None = None,
    hop_sec: float = 0.25,
    write_gated: bool = True,
    gate_categories: list[str] | None = None,
    full_scan: bool = False,
) -> EventResult:
    """
    Run frozen context detection.

    Default (``sed_enabled``): frame-level SED posteriors → hysteresis decoding →
    spectral-flux onset refinement → events.json → optional gated_audio.wav.

    Legacy path (``full_scan`` or ``sed_enabled: false``): sparse onset proposals →
    window tagging → frozen fusion. Kept for comparison; its onsets are only as
    precise as the 1 s classifier window.
    """
    video_path = Path(video_path)
    source_wav = Path(source_wav)
    taxonomy = load_taxonomy(taxonomy_path)
    window_sec = window_sec if window_sec is not None else taxonomy.proposal_window_sec
    gate_cats = resolve_gate_categories(taxonomy, gate_categories)

    tokens_data = tokenize_video_audio(video_path, source_wav, timeline_hz=taxonomy.timeline_hz)

    if taxonomy.sed_enabled and not full_scan:
        return _detect_via_frame_sed(
            tokens_data=tokens_data,
            source_wav=source_wav,
            video_path=video_path,
            taxonomy=taxonomy,
            gate_cats=gate_cats,
            output_dir=output_dir,
            write_gated=write_gated,
        )

    if full_scan:
        proposal_windows = None
        encoder_scores = run_encoder_pass(
            video_path,
            tokens_data.audio_16k,
            tokens_data.video_frames,
            window_sec=max(0.75, window_sec),
            hop_sec=hop_sec,
            duration_sec=tokens_data.duration_sec,
            full_scan=True,
        )
    else:
        proposal_windows = propose_all_windows(source_wav, taxonomy)
        encoder_scores = run_encoder_pass(
            video_path,
            tokens_data.audio_16k,
            tokens_data.video_frames,
            window_sec=window_sec,
            hop_sec=hop_sec,
            duration_sec=tokens_data.duration_sec,
            proposal_windows=proposal_windows,
            timeline_hz=taxonomy.timeline_hz,
        )

    tokens = symbolic_tokens_from_scores(encoder_scores, taxonomy=taxonomy)

    events = fuse_events(tokens, encoder_scores, taxonomy=taxonomy)
    events = refine_event_timing(events, source_wav, taxonomy)
    # Onset snap can collapse late rumble + early muzzle onto the same peak
    events = dedupe_events_by_peak(events)
    # Cannon in engine bed: AST often says vehicle; flux + explosion score recovers shots
    events = promote_impulsive_transients(
        events, source_wav, encoder_scores, taxonomy
    )
    # SED/refine often sit on a decay bump; snap back to the muzzle in-window
    events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
    events = align_impulsive_events_to_flashes(events, video_path, taxonomy)
    # Picture flash can lead the boom by ~0.2 s; snap again onto the sharp attack
    events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
    # Peaks are on attacks/flashes now; only the spans need rebuilding around them
    events = refine_event_timing(
        events, source_wav, taxonomy, relocate_impulsive_peaks=False
    )
    events = dedupe_events_by_peak(events)
    # One accent per blast: a shot plus a bump in its own decay is one bang
    events = suppress_impulsive_overlaps(events, source_wav, taxonomy)
    # Collapse fragmented vehicle chips into longer rumble spans
    events = merge_sustained_events(events, taxonomy)
    # Keep loud rumble islands; do not re-merge (that glues bursts across quiet gaps)
    gate_report: dict = {}
    events = filter_sustained_rumble_bursts(
        events, source_wav, taxonomy, report=gate_report
    )
    events = dedupe_events_by_peak(events)

    return _finalize(
        events,
        source_wav=source_wav,
        taxonomy=taxonomy,
        gate_cats=gate_cats,
        output_dir=output_dir,
        write_gated=write_gated,
        detector_info={"mode": "window_tagging", "use_video": taxonomy.use_video},
        sustained_gate=gate_report,
    )


def _finalize(
    events: list[DetectedEvent],
    *,
    source_wav: Path,
    taxonomy,
    gate_cats: list[str],
    output_dir: str | Path | None,
    write_gated: bool,
    detector_info: dict | None = None,
    sustained_gate: dict | None = None,
) -> EventResult:
    """Build EventResult and write events.json / gated audio."""
    events = measure_impulsive_attacks(events, source_wav, taxonomy)
    gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_cats)
    result = EventResult(
        events=events,
        no_events_detected=len(events) == 0,
        no_haptic_events=len(gate_events) == 0,
        timeline_hz=taxonomy.timeline_hz,
        gate_categories_used=gate_cats,
        detector_info=detector_info or {"mode": "window_tagging", "use_video": taxonomy.use_video},
        sustained_gate=sustained_gate or {},
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        events_json = output_dir / EVENTS_JSON_NAME
        events_json.write_text(
            json.dumps(result.to_dict(output_dir=output_dir), indent=2),
            encoding="utf-8",
        )
        result.events_json = events_json

        if write_gated and gate_events:
            gated = output_dir / GATED_AUDIO_NAME
            apply_gate(
                source_wav,
                gated,
                events,
                taxonomy=taxonomy,
                gate_categories=gate_cats,
            )
            result.gated_wav = gated
            result.haptic_outputs["gated_audio"] = str(gated)

    return result


def _detect_via_frame_sed(
    *,
    tokens_data,
    source_wav: Path,
    video_path: Path,
    taxonomy,
    gate_cats: list[str],
    output_dir: str | Path | None,
    write_gated: bool,
) -> EventResult:
    """
    Frame-level SED path (default).

    Per-category posteriors on a 100 ms grid (10 ms with PANNs) are decoded with
    median filtering + hysteresis, so onsets come from the model's time axis
    instead of one score per 1 s window. Spectral-flux refinement then sharpens
    impulsive onsets to sample accuracy.

    Video is not fused here on purpose — see ``use_video`` in taxonomy.yaml.
    """
    frames = compute_frame_posteriors(tokens_data.audio_16k, taxonomy)
    encoder_scores = posteriors_to_encoder_scores(frames, taxonomy)

    events = events_from_frame_posteriors(frames, taxonomy)
    events = split_impulsive_by_posterior_peaks(events, frames, taxonomy)
    events = refine_event_timing(events, source_wav, taxonomy)
    events = dedupe_events_by_peak(events)
    events = promote_impulsive_transients(events, source_wav, encoder_scores, taxonomy)
    # SED/refine often sit on a decay bump; snap back to the muzzle in-window
    events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
    events = align_impulsive_events_to_flashes(events, video_path, taxonomy)
    # Picture flash can lead the boom by ~0.2 s; snap again onto the sharp attack
    events = snap_impulsive_peaks_to_attacks(events, source_wav, taxonomy)
    # Peaks are on attacks/flashes now; only the spans need rebuilding around them
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

    return _finalize(
        events,
        source_wav=source_wav,
        taxonomy=taxonomy,
        gate_cats=gate_cats,
        output_dir=output_dir,
        write_gated=write_gated,
        sustained_gate=gate_report,
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
