"""Fusion detector: audio branch + video branch, fused, then post-processed.

The audio branch decodes frame-level sound-event posteriors (PANNs, or dense AST
as fallback) and sharpens impulsive onsets with spectral flux. The video branch
finds orange muzzle/explosion flashes and, optionally, labels scenes with
Qwen2.5-VL. ``taxonomy.yaml`` in this folder defines the categories and every
threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haptic_gt.core.contracts import DetectionResult, Event

from .branches import VISUAL_CONTEXT_NAME
from .detector import EVENTS_JSON_NAME, detect_events
from .frozen_fusion import DetectedEvent
from .manual_events import vehicle_events_from_peaks
from .mask import resolve_gate_categories
from .taxonomy import Taxonomy, load_taxonomy
from .visual_scenes import default_use_qwen

DETECTOR_EVENTS_NAME = "detector_events.json"


def replace_vehicle_with_manual_peaks(
    events: list[DetectedEvent],
    peaks_sec: list[float],
    taxonomy: Taxonomy,
) -> list[DetectedEvent]:
    """Keep impulsive auto events; replace vehicle spans with hand-marked rumble peaks."""
    kept = [ev for ev in events if ev.category != "vehicle"]
    kept.extend(vehicle_events_from_peaks(peaks_sec, taxonomy=taxonomy))
    kept.sort(key=lambda e: e.start_sec)
    return kept


def to_event(ev: DetectedEvent, taxonomy: Taxonomy, gate_categories: list[str]) -> Event:
    cat = taxonomy.categories.get(ev.category)
    extra: dict[str, Any] = {
        "context_token": ev.context_token,
        "audio_score": ev.audio_score,
        "video_score": ev.video_score,
    }
    if ev.attack_rel_max is not None:
        extra["attack_rel_max"] = ev.attack_rel_max
    if ev.attack_prominence is not None:
        extra["attack_prominence"] = ev.attack_prominence
    if ev.visual_categories:
        extra["visual_categories"] = list(ev.visual_categories)
    return Event(
        category=ev.category,
        label=ev.label,
        start_sec=ev.start_sec,
        peak_sec=ev.peak_sec,
        end_sec=ev.end_sec,
        confidence=ev.confidence,
        impulsive=bool(cat is not None and cat.impulsive),
        in_gate=ev.category in gate_categories,
        sources=list(ev.sources),
        extra=extra,
    )


class Component:
    def __init__(
        self,
        taxonomy_path: str | None = None,
        gate_categories: list[str] | None = None,
        use_qwen: bool | None = None,
        full_scan: bool = False,
        manual_rumble_peaks: list[float] | None = None,
    ):
        """
        ``gate_categories``: categories that drive the haptics (default: those with
        ``include_in_haptic_gate`` in the taxonomy). ``use_qwen``: scene labels in
        the video branch (default: on only with ``visual_scenes_enabled`` and a GPU
        of at least ``visual_scenes_min_gpu_gb``, so off on a T4). ``manual_rumble_peaks``:
        hand-marked rumble times that replace auto-detected ``vehicle`` spans.
        """
        self.taxonomy_path = taxonomy_path
        self.gate_categories = gate_categories
        self.use_qwen = use_qwen
        self.full_scan = full_scan
        self.manual_rumble_peaks = manual_rumble_peaks

    def detect(self, video_path: Path | None, source_wav: Path, workdir: Path) -> DetectionResult:
        taxonomy = load_taxonomy(self.taxonomy_path)
        gate_cats = resolve_gate_categories(taxonomy, self.gate_categories)
        report: dict[str, Any] = {
            "gate_categories_used": gate_cats,
            "timeline_hz": taxonomy.timeline_hz,
            "categories": {
                name: {"impulsive": cat.impulsive, "in_gate": name in gate_cats}
                for name, cat in taxonomy.categories.items()
            },
        }
        if video_path is None:
            report = {"detector": {"mode": "skipped", "reason": "needs a video input"}, **report}
            return DetectionResult(events=[], report=report)

        if self.use_qwen is None:
            use_qwen, qwen_reason = default_use_qwen(taxonomy)
        else:
            use_qwen, qwen_reason = self.use_qwen, "set in pipeline params"
        print(f"Qwen scene labels: {'on' if use_qwen else 'off'} ({qwen_reason})")

        result = detect_events(
            video_path,
            source_wav,
            workdir,
            taxonomy_path=self.taxonomy_path,
            write_gated=False,
            gate_categories=self.gate_categories,
            full_scan=self.full_scan,
            use_qwen=use_qwen,
        )
        events = result.events
        if self.manual_rumble_peaks:
            events = replace_vehicle_with_manual_peaks(events, self.manual_rumble_peaks, taxonomy)

        report = {
            "detector": {**result.detector_info, "qwen_scenes": use_qwen, "qwen_reason": qwen_reason},
            "sustained_gate": result.sustained_gate,
            **report,
        }
        if result.fusion:
            report["fusion"] = result.fusion

        artifacts: dict[str, Path] = {}
        native = workdir / EVENTS_JSON_NAME
        if native.is_file():
            artifacts["detector_events"] = native.replace(workdir / DETECTOR_EVENTS_NAME)
        visual = workdir / VISUAL_CONTEXT_NAME
        if visual.is_file():
            artifacts["visual_context"] = visual

        return DetectionResult(
            events=[to_event(ev, taxonomy, gate_cats) for ev in events],
            report=report,
            artifacts=artifacts,
        )
