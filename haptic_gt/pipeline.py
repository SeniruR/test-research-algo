"""End-to-end pipeline aligned with Sound2Hap signal processing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any

from haptic_gt.algorithms import freq_shift, haptic_gen, percept, pitch_match, rule_based
from haptic_gt.audio_io import INPUT_SR, VIB_SR, extract_audio_from_video, prepare_source_wav
from haptic_gt.context import detect_events
from haptic_gt.context.detector import EVENTS_JSON_NAME, GATED_AUDIO_NAME, EventResult
from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.manual_events import events_from_manual, vehicle_events_from_peaks
from haptic_gt.context.mask import apply_gate, events_for_haptic_gate, resolve_gate_categories
from haptic_gt.context.taxonomy import load_taxonomy
from haptic_gt.haptic_synthesis import ContinuousProfile, stitch_algorithm_output


def _replace_vehicle_with_manual_peaks(
    events: list[DetectedEvent],
    peaks_sec: list[float],
    taxonomy,
) -> list[DetectedEvent]:
    """Keep impulsive auto events; replace vehicle spans with hand-marked rumble peaks."""
    kept = []
    for ev in events:
        cat = taxonomy.categories.get(ev.category)
        if cat is not None and not cat.impulsive and ev.category == "vehicle":
            continue
        if ev.category == "vehicle":
            continue
        kept.append(ev)
    kept.extend(vehicle_events_from_peaks(peaks_sec, taxonomy=taxonomy))
    kept.sort(key=lambda e: e.start_sec)
    return kept

OUTPUT_NAMES = {
    "source_audio": "source_audio.wav",
    "gated_audio": "gated_audio.wav",
    "events_json": "events.json",
    "algorithm_a_perception_mapping": "algorithm_a_perception_mapping.wav",
    "algorithm_b_frequency_shifting": "algorithm_b_frequency_shifting.wav",
    "algorithm_c_pitch_matching": "algorithm_c_pitch_matching.wav",
    "algorithm_d_haptic_gen": "algorithm_d_haptic_gen.wav",
    "algorithm_e_rule_based": "algorithm_e_rule_based.wav",
    "algorithm_e_rule_based_json": "algorithm_e_rule_based.json",
}


@dataclass
class CandidateTracks:
    """Paths to source audio, Sound2Hap A–D, and rule-based E."""

    source_wav: Path
    algorithm_a: Path | None
    algorithm_b: Path | None
    algorithm_c: Path | None
    algorithm_d: Path | None
    algorithm_e: Path | None
    algorithm_e_json: Path | None
    output_dir: Path
    haptic_input_wav: Path | None
    input_sample_rate: int = INPUT_SR
    output_sample_rate: int = VIB_SR
    pitch_match_info: dict | None = None
    events: list[DetectedEvent] | None = None
    events_json: Path | None = None
    no_events_detected: bool = False
    no_haptic_events: bool = False
    gate_categories_used: list[str] = field(default_factory=list)

    def save_all(self) -> dict[str, Path]:
        out: dict[str, Path] = {"source_audio": self.source_wav}
        if self.algorithm_a and self.algorithm_a.exists():
            out["algorithm_a_perception_mapping"] = self.algorithm_a
        if self.algorithm_b and self.algorithm_b.exists():
            out["algorithm_b_frequency_shifting"] = self.algorithm_b
        if self.algorithm_c and self.algorithm_c.exists():
            out["algorithm_c_pitch_matching"] = self.algorithm_c
        if self.algorithm_d and self.algorithm_d.exists():
            out["algorithm_d_haptic_gen"] = self.algorithm_d
        if self.algorithm_e and self.algorithm_e.exists():
            out["algorithm_e_rule_based"] = self.algorithm_e
        if self.algorithm_e_json and self.algorithm_e_json.exists():
            out["algorithm_e_rule_based_json"] = self.algorithm_e_json
        if self.events_json and self.events_json.exists():
            out["events_json"] = self.events_json
        if (
            self.haptic_input_wav is not None
            and self.haptic_input_wav.exists()
            and self.haptic_input_wav != self.source_wav
        ):
            out["gated_audio"] = self.haptic_input_wav
        return out


def _update_events_json_haptics(
    events_json: Path,
    haptic_paths: dict[str, str],
) -> None:
    if not events_json.exists():
        return
    payload = json.loads(events_json.read_text(encoding="utf-8"))
    existing = payload.get("haptic_outputs", {})
    existing.update(haptic_paths)
    payload["haptic_outputs"] = existing
    events_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_candidate_tracks(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    from_video: bool = True,
    content_type: str = "game",
    enable_context_detection: bool = True,
    taxonomy_path: str | Path | None = None,
    gate_categories: list[str] | None = None,
    continuous_haptics: bool = True,
    continuous_profile: ContinuousProfile | None = None,
    manual_events: list[dict[str, Any]] | dict[str, Any] | str | Path | None = None,
    manual_rumble_peaks: list[float] | None = None,
) -> CandidateTracks:
    """
    Run context detection (optional), Sound2Hap A–D, and rule-based E.

    When gate-eligible events are detected, A–D are stitched onto a full-length
    timeline (one WAV per algorithm). When none match gate_categories, A–D are
    skipped. Rule-based E always runs on the ungated mix (plus video frames
    when ``from_video`` is true).

    With `continuous_haptics` on, each algorithm also renders the full clip as a
    low-level continuous layer underneath the event accents, so sustained sounds
    (rumble, rain, engines) keep vibrating instead of leaving silent gaps.

    Pass ``manual_events`` (list of dicts, single event dict, or path to
    events.json) to skip AST/ViViT and trust hand-labeled start/peak/end times.

    Pass ``manual_rumble_peaks`` to replace auto ``vehicle`` events with your
    marked rumble times (keeps auto gunshot/explosion). Use when calib shows
    RMS rise cannot separate true rumbles from engine-bed false positives.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(taxonomy_path)
    gate_cats = resolve_gate_categories(taxonomy, gate_categories)

    source_wav = output_dir / OUTPUT_NAMES["source_audio"]
    if from_video:
        extract_audio_from_video(input_path, source_wav, sr=INPUT_SR)
    else:
        prepare_source_wav(input_path, source_wav, sr=INPUT_SR)

    events: list[DetectedEvent] | None = None
    events_json_path: Path | None = None
    no_events_detected = False
    no_haptic_events = False
    haptic_input: Path | None = None
    gate_events: list[DetectedEvent] = []

    out_a = output_dir / OUTPUT_NAMES["algorithm_a_perception_mapping"]
    out_b = output_dir / OUTPUT_NAMES["algorithm_b_frequency_shifting"]
    out_c = output_dir / OUTPUT_NAMES["algorithm_c_pitch_matching"]
    out_d = output_dir / OUTPUT_NAMES["algorithm_d_haptic_gen"]
    out_e = output_dir / OUTPUT_NAMES["algorithm_e_rule_based"]
    out_e_json = output_dir / OUTPUT_NAMES["algorithm_e_rule_based_json"]

    if manual_events is not None:
        events = events_from_manual(manual_events, taxonomy)
        gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_cats)
        no_events_detected = len(events) == 0
        no_haptic_events = len(gate_events) == 0
        result = EventResult(
            events=events,
            no_events_detected=no_events_detected,
            no_haptic_events=no_haptic_events,
            timeline_hz=taxonomy.timeline_hz,
            gate_categories_used=gate_cats,
        )
        events_json_path = output_dir / EVENTS_JSON_NAME
        events_json_path.write_text(
            json.dumps(result.to_dict(output_dir=output_dir), indent=2),
            encoding="utf-8",
        )
        result.events_json = events_json_path
        if gate_events:
            gated = output_dir / GATED_AUDIO_NAME
            apply_gate(
                source_wav,
                gated,
                events,
                taxonomy=taxonomy,
                gate_categories=gate_cats,
            )
            haptic_input = gated
            result.gated_wav = gated
            result.haptic_outputs["gated_audio"] = str(gated)
            # Refresh JSON so gated path is recorded
            events_json_path.write_text(
                json.dumps(result.to_dict(output_dir=output_dir), indent=2),
                encoding="utf-8",
            )
    elif enable_context_detection and from_video:
        event_result = detect_events(
            input_path,
            source_wav,
            output_dir,
            taxonomy_path=taxonomy_path,
            write_gated=True,
            gate_categories=gate_categories,
        )
        events = event_result.events
        events_json_path = event_result.events_json
        if manual_rumble_peaks:
            events = _replace_vehicle_with_manual_peaks(
                events or [], manual_rumble_peaks, taxonomy
            )
            # Rewrite events.json + gated audio with replaced vehicle spans
            gate_events = events_for_haptic_gate(events, taxonomy, gate_categories=gate_cats)
            result = EventResult(
                events=events,
                no_events_detected=len(events) == 0,
                no_haptic_events=len(gate_events) == 0,
                timeline_hz=taxonomy.timeline_hz,
                gate_categories_used=gate_cats,
            )
            events_json_path = output_dir / EVENTS_JSON_NAME
            events_json_path.write_text(
                json.dumps(result.to_dict(output_dir=output_dir), indent=2),
                encoding="utf-8",
            )
            if gate_events:
                gated = output_dir / GATED_AUDIO_NAME
                apply_gate(
                    source_wav,
                    gated,
                    events,
                    taxonomy=taxonomy,
                    gate_categories=gate_cats,
                )
                haptic_input = gated
            no_events_detected = len(events) == 0
            no_haptic_events = len(gate_events) == 0
        else:
            no_events_detected = event_result.no_events_detected
            no_haptic_events = event_result.no_haptic_events
            gate_events = events_for_haptic_gate(
                events or [], taxonomy, gate_categories=gate_cats
            )
            if event_result.gated_wav is not None and event_result.gated_wav.exists():
                haptic_input = event_result.gated_wav
    elif enable_context_detection:
        events_json_path = output_dir / OUTPUT_NAMES["events_json"]
        payload = {
            "no_events_detected": True,
            "no_haptic_events": True,
            "gate_categories_used": gate_cats,
            "timeline_hz": taxonomy.timeline_hz,
            "haptic_outputs": {},
            "events": [],
        }
        events_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        no_events_detected = True
        no_haptic_events = True
    else:
        gate_events = []

    profile = continuous_profile or ContinuousProfile(enabled=continuous_haptics)

    pitch_info = None
    if gate_events:
        stitch_algorithm_output(
            source_wav,
            gate_events,
            out_a,
            percept.process_file,
            process_kwargs={"content": content_type},
            continuous=profile,
            taxonomy=taxonomy,
        )
        stitch_algorithm_output(
            source_wav,
            gate_events,
            out_b,
            freq_shift.process_file,
            continuous=profile,
            taxonomy=taxonomy,
        )
        stitch_algorithm_output(
            source_wav,
            gate_events,
            out_c,
            pitch_match.process_file,
            continuous=profile,
            taxonomy=taxonomy,
        )
        stitch_algorithm_output(
            source_wav,
            gate_events,
            out_d,
            haptic_gen.process_file,
            continuous=profile,
            taxonomy=taxonomy,
        )

        if events_json_path is not None:
            _update_events_json_haptics(
                events_json_path,
                {
                    "algorithm_a": OUTPUT_NAMES["algorithm_a_perception_mapping"],
                    "algorithm_b": OUTPUT_NAMES["algorithm_b_frequency_shifting"],
                    "algorithm_c": OUTPUT_NAMES["algorithm_c_pitch_matching"],
                    "algorithm_d": OUTPUT_NAMES["algorithm_d_haptic_gen"],
                },
            )

    rule_based.process_file(
        source_wav,
        out_e,
        video_path=input_path if from_video else None,
        json_path=out_e_json,
    )
    if events_json_path is not None:
        _update_events_json_haptics(
            events_json_path,
            {"algorithm_e": OUTPUT_NAMES["algorithm_e_rule_based"]},
        )

    return CandidateTracks(
        source_wav=source_wav,
        algorithm_a=out_a if gate_events else None,
        algorithm_b=out_b if gate_events else None,
        algorithm_c=out_c if gate_events else None,
        algorithm_d=out_d if gate_events else None,
        algorithm_e=out_e,
        algorithm_e_json=out_e_json,
        output_dir=output_dir,
        haptic_input_wav=haptic_input,
        pitch_match_info=pitch_info,
        events=events,
        events_json=events_json_path,
        no_events_detected=no_events_detected,
        no_haptic_events=no_haptic_events,
        gate_categories_used=gate_cats,
    )
