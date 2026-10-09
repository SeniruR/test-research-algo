"""Run the configured detector and generators over one clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haptic_gt.core.audio_io import INPUT_SR, VIB_SR, extract_audio_from_video, prepare_source_wav
from haptic_gt.core.config import PipelineConfig, load_pipeline_config
from haptic_gt.core.contracts import DetectionResult, Event, GeneratorContext
from haptic_gt.core.gating import apply_gate
from haptic_gt.core.packaging import (
    EVENTS_JSON_NAME,
    GATED_AUDIO_NAME,
    SOURCE_AUDIO_NAME,
    write_events_json,
)
from haptic_gt.core.registry import load_detector, load_generators
from haptic_gt.core.synthesis import stitch_algorithm_output

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


@dataclass
class RunResult:
    output_dir: Path
    source_wav: Path
    #: Rendered haptic WAVs keyed by the generator's output_name.
    outputs: dict[str, Path] = field(default_factory=dict)
    detection: DetectionResult | None = None
    events_json: Path | None = None
    gated_wav: Path | None = None
    #: Whatever each generator returned, keyed by component name.
    generator_reports: dict[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> list[Event]:
        return self.detection.events if self.detection else []

    @property
    def gate_events(self) -> list[Event]:
        return self.detection.gate_events if self.detection else []

    @property
    def no_events_detected(self) -> bool:
        return not self.events

    @property
    def no_haptic_events(self) -> bool:
        return not self.gate_events

    def save_all(self) -> dict[str, Path]:
        """Every artifact worth keeping, keyed for write_candidate_archive."""
        out: dict[str, Path] = {"source_audio": self.source_wav}
        for name, path in self.outputs.items():
            if path.exists():
                out[Path(name).stem] = path
        if self.events_json is not None and self.events_json.exists():
            out["events_json"] = self.events_json
        if self.gated_wav is not None and self.gated_wav.exists():
            out["gated_audio"] = self.gated_wav
        return out


def _remove_if_empty(directory: Path) -> None:
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()


def run_pipeline(
    config: PipelineConfig | None,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    from_video: bool | None = None,
) -> RunResult:
    """
    Extract audio, detect events, gate, and render every configured generator.

    Generators with ``input: events`` only run when the detector marked at least
    one event for the haptic gate; ``full_mix`` generators always run. With no
    detector configured, only ``full_mix`` generators produce output.
    """
    config = config or load_pipeline_config()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if from_video is None:
        from_video = input_path.suffix.lower() in VIDEO_SUFFIXES

    detector = load_detector(config.detector)
    generators = load_generators(config.generators)

    source_wav = output_dir / SOURCE_AUDIO_NAME
    if from_video:
        extract_audio_from_video(input_path, source_wav, sr=INPUT_SR)
    else:
        prepare_source_wav(input_path, source_wav, sr=INPUT_SR)
    video_path = input_path if from_video else None

    result = RunResult(output_dir=output_dir, source_wav=source_wav)
    haptic_outputs: dict[str, str] = {}

    if detector is not None:
        workdir = output_dir / "debug" / "context_detector"
        workdir.mkdir(parents=True, exist_ok=True)
        result.detection = detector.detect(video_path, source_wav, workdir)
        _remove_if_empty(workdir)
        if result.gate_events:
            result.gated_wav = apply_gate(
                source_wav, output_dir / GATED_AUDIO_NAME, result.gate_events
            )
            haptic_outputs["gated_audio"] = GATED_AUDIO_NAME
        result.events_json = write_events_json(
            output_dir / EVENTS_JSON_NAME, result.detection, haptic_outputs=haptic_outputs
        )

    synth = config.synthesis
    for gen in generators:
        name = gen.component_name
        out_path = output_dir / gen.output_name
        debug_dir = output_dir / "debug" / name
        debug_dir.mkdir(parents=True, exist_ok=True)
        ctx = GeneratorContext(
            source_wav=source_wav,
            video_path=video_path,
            debug_dir=debug_dir,
            input_sr=INPUT_SR,
            output_sr=VIB_SR,
        )

        if gen.input == "events":
            if not result.gate_events:
                _remove_if_empty(debug_dir)
                continue
            reports: list[Any] = []

            def _process(clip_in, clip_out, _gen=gen, _ctx=ctx, _reports=reports):
                _reports.append(_gen.generate(Path(clip_in), Path(clip_out), _ctx))

            stitch_algorithm_output(
                source_wav,
                result.gate_events,
                out_path,
                _process,
                input_sr=INPUT_SR,
                output_sr=VIB_SR,
                continuous=synth.continuous,
                sustained_mask_min_events=synth.sustained_mask_min_events,
                sustained_mask_min_coverage=synth.sustained_mask_min_coverage,
            )
            report = next((r for r in reversed(reports) if r), None)
        else:
            report = gen.generate(source_wav, out_path, ctx)

        _remove_if_empty(debug_dir)
        result.outputs[gen.output_name] = out_path
        if report:
            result.generator_reports[name] = report
        haptic_outputs[f"algorithm_{gen.slot}"] = gen.output_name

    if result.detection is not None:
        write_events_json(
            output_dir / EVENTS_JSON_NAME, result.detection, haptic_outputs=haptic_outputs
        )
    _remove_if_empty(output_dir / "debug")
    return result
