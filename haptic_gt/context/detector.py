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
from haptic_gt.context.taxonomy import load_taxonomy
from haptic_gt.context.tokenization import tokenize_video_audio

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
            event_rows.append(
                {
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
            )

        haptic_out = {
            key: _rel(val) for key, val in self.haptic_outputs.items() if val
        }

        return {
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
    Run frozen context detection: propose onsets → classify windows → fuse.

    Phase 1: tokenization → onset proposals → targeted AST/ViViT
    Phase 2: frozen fusion → events.json → optional gated_audio.wav
    """
    video_path = Path(video_path)
    source_wav = Path(source_wav)
    taxonomy = load_taxonomy(taxonomy_path)
    window_sec = window_sec if window_sec is not None else taxonomy.proposal_window_sec
    gate_cats = resolve_gate_categories(taxonomy, gate_categories)

    tokens_data = tokenize_video_audio(video_path, source_wav, timeline_hz=taxonomy.timeline_hz)

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

    gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_cats)
    result = EventResult(
        events=events,
        no_events_detected=len(events) == 0,
        no_haptic_events=len(gate_events) == 0,
        timeline_hz=taxonomy.timeline_hz,
        gate_categories_used=gate_cats,
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
