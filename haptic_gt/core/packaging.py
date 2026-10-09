"""events.json and the downloadable candidate archive."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from haptic_gt.core.audio_io import remux_to_seekable_mp4
from haptic_gt.core.contracts import DetectionResult

SOURCE_AUDIO_NAME = "source_audio.wav"
GATED_AUDIO_NAME = "gated_audio.wav"
EVENTS_JSON_NAME = "events.json"

#: Archive folder for core artifacts; anything else that is not a playback WAV
#: goes under debug/pipeline/.
DEBUG_OUTPUT_DIRECTORIES = {
    "source_audio": "debug/audio_preparation",
    "gated_audio": "debug/context_detector",
    "events_json": "debug/context_detector",
}


def build_events_payload(
    detection: DetectionResult,
    *,
    haptic_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(detection.report)
    payload["no_events_detected"] = not detection.events
    payload["no_haptic_events"] = not detection.gate_events
    payload["haptic_outputs"] = dict(haptic_outputs or {})
    payload["events"] = [
        ev.to_row(f"event_{i:03d}") for i, ev in enumerate(detection.events, start=1)
    ]
    return payload


def write_events_json(
    path: str | Path,
    detection: DetectionResult,
    *,
    haptic_outputs: dict[str, str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_events_payload(detection, haptic_outputs=haptic_outputs)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _is_playback(key: str, path: Path) -> bool:
    return key not in DEBUG_OUTPUT_DIRECTORIES and path.suffix.lower() == ".wav"


def write_candidate_archive(
    saved_outputs: dict[str, Path],
    video_path: str | Path,
    archive_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Package app-ready video/WAVs at the root and other artifacts under debug/."""
    video_path = Path(video_path)
    archive_path = Path(archive_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    playback_outputs = {
        key: Path(path)
        for key, path in saved_outputs.items()
        if Path(path).is_file() and _is_playback(key, Path(path))
    }
    if not playback_outputs:
        raise ValueError("No generated haptic WAV files are available to package")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archived_paths = {video_path.resolve()}
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive, \
            tempfile.TemporaryDirectory() as tmp:
        try:
            app_video = remux_to_seekable_mp4(video_path, Path(tmp) / "video.mp4")
        except RuntimeError as exc:
            warnings.warn(f"Packaging the original video; remux failed: {exc}")
            app_video = video_path
        archive.write(app_video, f"video{app_video.suffix.lower()}")
        for path in playback_outputs.values():
            archive.write(path, path.name)
            archived_paths.add(path.resolve())

        for key, raw_path in saved_outputs.items():
            path = Path(raw_path)
            if key in playback_outputs or not path.is_file():
                continue
            directory = DEBUG_OUTPUT_DIRECTORIES.get(key, f"debug/pipeline/{key}")
            archive.write(path, f"{directory}/{path.name}")
            archived_paths.add(path.resolve())

        if output_dir is not None:
            output_root = Path(output_dir)
            if output_root.is_dir():
                for path in sorted(output_root.rglob("*")):
                    if not path.is_file() or path.resolve() in archived_paths:
                        continue
                    relative_path = path.relative_to(output_root).as_posix()
                    if not relative_path.startswith("debug/"):
                        relative_path = f"debug/pipeline/{relative_path}"
                    archive.write(path, relative_path)

    return archive_path
