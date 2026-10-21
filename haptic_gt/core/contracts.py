"""Data passed between the core and pluggable components.

Two component kinds exist:

- A **context detector** finds events in a clip. It owns its own taxonomy and
  marks each event as ``impulsive`` (accent at the peak) or sustained (follows
  its span), and whether it belongs in the haptic gate.
- A **haptic generator** turns audio into an 8 kHz vibration WAV. Generators
  with ``input = "events"`` see one short clip per gated event and the core
  stitches the results onto the timeline; ``input = "full_mix"`` generators run
  once on the whole, ungated source audio.

Components only depend on this module (and other ``haptic_gt.core`` helpers),
never on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

GeneratorInput = Literal["events", "full_mix"]


@dataclass
class Event:
    category: str
    label: str
    start_sec: float
    peak_sec: float
    end_sec: float
    confidence: float
    impulsive: bool = False
    in_gate: bool = False
    sources: list[str] = field(default_factory=list)
    #: Detector-specific fields, written to events.json as-is.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self, event_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "event_id": event_id,
            "category": self.category,
            "label": self.label,
            "start_sec": round(self.start_sec, 3),
            "peak_sec": round(self.peak_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "confidence": round(self.confidence, 4),
            "impulsive": self.impulsive,
        }
        row.update(self.extra)
        row["sources"] = list(self.sources)
        row["included_in_gate"] = self.in_gate
        return row


@dataclass
class DetectionResult:
    events: list[Event] = field(default_factory=list)
    #: Merged into the top level of events.json (e.g. "detector", "fusion").
    report: dict[str, Any] = field(default_factory=dict)
    #: Extra files the detector wrote, by name.
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def gate_events(self) -> list[Event]:
        return [e for e in self.events if e.in_gate]


@dataclass
class GeneratorContext:
    source_wav: Path
    video_path: Path | None
    #: Folder reserved for this generator's auxiliary files.
    debug_dir: Path
    input_sr: int
    output_sr: int


@runtime_checkable
class ContextDetector(Protocol):
    def detect(
        self,
        video_path: Path | None,
        source_wav: Path,
        workdir: Path,
    ) -> DetectionResult: ...


@runtime_checkable
class HapticGenerator(Protocol):
    #: Single letter used by the Android app to pick a compare slot (a-e).
    slot: str
    #: File name of the rendered WAV, e.g. "algorithm_a_perception_mapping.wav".
    output_name: str
    input: GeneratorInput

    def generate(
        self,
        in_wav: Path,
        out_wav: Path,
        ctx: GeneratorContext,
    ) -> dict[str, Any] | None: ...
