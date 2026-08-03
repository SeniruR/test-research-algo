"""Debug helpers: event timing table + per-event audio/metadata extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.audio_io import INPUT_SR


@dataclass
class EventDebugRow:
    event_id: str
    category: str
    label: str
    start_sec: float
    peak_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    audio_score: float | None
    video_score: float | None
    sources: list[str]
    included_in_gate: bool
    clip_wav: str | None = None
    # Filled by user during diagnosis (optional)
    actual_peak_sec: float | None = None
    timing_note: str | None = None


def load_events_payload(events_json: str | Path) -> dict:
    path = Path(events_json)
    return json.loads(path.read_text(encoding="utf-8"))


def events_to_debug_rows(payload: dict) -> list[EventDebugRow]:
    rows: list[EventDebugRow] = []
    for i, ev in enumerate(payload.get("events", []), start=1):
        event_id = str(ev.get("event_id") or f"event_{i:03d}")
        start = float(ev.get("start_sec", 0.0))
        peak = float(ev.get("peak_sec", start))
        end = float(ev.get("end_sec", peak))
        rows.append(
            EventDebugRow(
                event_id=event_id,
                category=str(ev.get("category", "")),
                label=str(ev.get("label", "")),
                start_sec=start,
                peak_sec=peak,
                end_sec=end,
                duration_sec=max(0.0, end - start),
                confidence=float(ev.get("confidence", 0.0)),
                audio_score=ev.get("audio_score"),
                video_score=ev.get("video_score"),
                sources=list(ev.get("sources") or []),
                included_in_gate=bool(ev.get("included_in_gate", False)),
            )
        )
    return rows


def format_events_table(rows: list[EventDebugRow]) -> str:
    """Plain-text timing table for Colab / terminal diagnosis."""
    if not rows:
        return "(no events)"

    header = (
        f"{'id':<12} {'cat':<14} {'start':>8} {'peak':>8} {'end':>8} "
        f"{'dur':>6} {'conf':>6} {'audio':>6} {'vid':>6} {'gate':>5} label"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        audio = f"{r.audio_score:.3f}" if r.audio_score is not None else "-"
        video = f"{r.video_score:.3f}" if r.video_score is not None else "-"
        lines.append(
            f"{r.event_id:<12} {r.category:<14} "
            f"{r.start_sec:8.3f} {r.peak_sec:8.3f} {r.end_sec:8.3f} "
            f"{r.duration_sec:6.3f} {r.confidence:6.3f} {audio:>6} {video:>6} "
            f"{'Y' if r.included_in_gate else 'N':>5} {r.label}"
        )
    return "\n".join(lines)


def extract_event_debug_clips(
    source_wav: str | Path,
    events_json: str | Path,
    output_dir: str | Path,
    *,
    sample_rate: int = INPUT_SR,
    pad_sec: float = 0.25,
) -> list[EventDebugRow]:
    """
    Write one WAV + JSON sidecar per event for accurate listening/diagnosis.

    Layout:
      debug_events/
        event_001_vehicle/
          clip.wav
          meta.json
        ...
        timing_table.txt
        events_debug.json
    """
    source_wav = Path(source_wav)
    events_json = Path(events_json)
    output_dir = Path(output_dir)
    debug_root = output_dir / "debug_events"
    debug_root.mkdir(parents=True, exist_ok=True)

    payload = load_events_payload(events_json)
    rows = events_to_debug_rows(payload)

    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    duration_sec = len(audio) / sr

    exported: list[EventDebugRow] = []
    for row in rows:
        folder = debug_root / f"{row.event_id}_{row.category}"
        folder.mkdir(parents=True, exist_ok=True)
        clip_path = folder / "clip.wav"

        s0 = max(0.0, row.start_sec - pad_sec)
        s1 = min(duration_sec, row.end_sec + pad_sec)
        i0 = max(0, int(s0 * sr))
        i1 = min(len(audio), int(s1 * sr))
        clip = audio[i0:i1]
        sf.write(clip_path, clip, sr, subtype="PCM_16")

        meta = asdict(row)
        meta.update(
            {
                "clip_wav": str(clip_path.relative_to(output_dir)),
                "clip_start_sec": round(s0, 3),
                "clip_end_sec": round(s1, 3),
                "pad_sec": pad_sec,
                "sample_rate": sr,
                # User fills these after listening / watching video:
                "actual_peak_sec": None,
                "timing_error_sec": None,
                "note": None,
            }
        )
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        row.clip_wav = meta["clip_wav"]
        exported.append(row)

    table = format_events_table(exported)
    (debug_root / "timing_table.txt").write_text(table + "\n", encoding="utf-8")

    summary = {
        "source_audio": str(source_wav.name),
        "gate_categories_used": payload.get("gate_categories_used", []),
        "instructions": (
            "Compare peak_sec to the real event in the video/audio. "
            "Fill actual_peak_sec and timing_error_sec (= detected_peak - actual_peak) "
            "in each meta.json or reply with a table."
        ),
        "events": [asdict(r) for r in exported],
    }
    (debug_root / "events_debug.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return exported


def print_event_timing_report(
    events_json: str | Path,
    *,
    source_wav: str | Path | None = None,
    output_dir: str | Path | None = None,
    extract_clips: bool = True,
) -> list[EventDebugRow]:
    """Print timing table and optionally extract per-event debug clips."""
    payload = load_events_payload(events_json)
    rows = events_to_debug_rows(payload)
    print("=== Detected event timing table ===")
    print(format_events_table(rows))
    print()
    print("Fill actual peaks like:")
    print("  event_001 actual_peak_sec=...  (or 'ok' if correct)")
    print("  event_004 actual_peak_sec=15.20")
    print()

    if extract_clips and source_wav is not None and output_dir is not None:
        exported = extract_event_debug_clips(source_wav, events_json, output_dir)
        print(f"Wrote debug clips under: {Path(output_dir) / 'debug_events'}")
        return exported
    return rows
