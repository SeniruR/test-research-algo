"""Tests for end-to-end pipeline helpers (no model downloads)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.haptic_synthesis import ContinuousProfile, stitch_algorithm_output
from haptic_gt.pipeline import OUTPUT_NAMES, generate_candidate_tracks


def _write_test_wav(path: Path, duration: float = 3.0, sr: int = 44100) -> None:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr, subtype="PCM_16")


def _active_fraction(path: Path, window_ms: int = 20) -> float:
    """Fraction of windows carrying any haptic energy."""
    audio, sr = sf.read(path, always_2d=False)
    per_window = max(1, (sr * window_ms) // 1000)
    n_windows = int(np.ceil(audio.size / per_window))
    padded = np.pad(audio, (0, n_windows * per_window - audio.size))
    peaks = np.max(np.abs(padded.reshape(n_windows, per_window)), axis=1)
    return float(np.mean(peaks > 1e-4))


def test_stitch_algorithm_output_full_length():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        out = td_path / "haptic.wav"
        _write_test_wav(source, duration=5.0)

        events = [
            DetectedEvent("gunshot", "Gunshot", 1.0, 1.2, 1.5, 0.9),
            DetectedEvent("gunshot", "Gunshot", 3.0, 3.2, 3.5, 0.9),
        ]
        stitch_algorithm_output(source, events, out, lambda i, o: _write_test_wav(o, 0.5))

        haptic, sr = sf.read(out)
        assert sr == 8000
        assert len(haptic) == int(round(5.0 * 8000))


def test_continuous_layer_fills_gaps_between_events():
    """The continuous layer should cover far more of the timeline than events alone."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        _write_test_wav(source, duration=6.0)

        events = [DetectedEvent("gunshot", "Gunshot", 1.0, 1.2, 1.5, 0.9)]

        def _passthrough(in_wav, out_wav):
            audio, sr = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, audio, sr, subtype="PCM_16")

        event_only = td_path / "event_only.wav"
        continuous = td_path / "continuous.wav"
        stitch_algorithm_output(
            source, events, event_only, _passthrough, continuous=ContinuousProfile(enabled=False)
        )
        stitch_algorithm_output(
            source, events, continuous, _passthrough, continuous=ContinuousProfile()
        )

        assert _active_fraction(event_only) < 0.25
        assert _active_fraction(continuous) > 0.60


def test_continuous_layer_keeps_events_loudest():
    """Event accents must stay above the continuous bed so hits remain distinct."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        out = td_path / "haptic.wav"
        _write_test_wav(source, duration=6.0)

        events = [DetectedEvent("gunshot", "Gunshot", 2.0, 2.2, 2.5, 0.9)]

        def _passthrough(in_wav, out_wav):
            audio, sr = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, audio, sr, subtype="PCM_16")

        profile = ContinuousProfile()
        stitch_algorithm_output(source, events, out, _passthrough, continuous=profile)

        haptic, sr = sf.read(out)
        near_event = np.max(np.abs(haptic[int(2.0 * sr) : int(2.6 * sr)]))
        bed = np.max(np.abs(haptic[int(4.0 * sr) : int(5.0 * sr)]))
        assert near_event > bed


def test_pipeline_skips_haptics_when_no_gate_events():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fake_video = td_path / "clip.mp4"
        fake_video.write_bytes(b"not-a-real-video")

        mock_events = [
            DetectedEvent("vehicle", "Vehicle", 0.5, 1.0, 1.5, 0.7),
        ]

        def _fake_extract(video_path, output_path, sr=44100):
            _write_test_wav(Path(output_path))
            return Path(output_path)

        with patch("haptic_gt.pipeline.extract_audio_from_video", side_effect=_fake_extract):
            with patch("haptic_gt.pipeline.detect_events") as mock_detect:
                from haptic_gt.context.detector import EventResult

                mock_detect.return_value = EventResult(
                    events=mock_events,
                    no_events_detected=False,
                    no_haptic_events=True,
                    gate_categories_used=["gunshot", "explosion"],
                )

                tracks = generate_candidate_tracks(
                    fake_video,
                    td_path,
                    from_video=True,
                    enable_context_detection=True,
                    gate_categories=["gunshot", "explosion"],
                )

        assert tracks.algorithm_a is None
        assert tracks.no_haptic_events is True


def test_output_names_layout():
    assert OUTPUT_NAMES["algorithm_a_perception_mapping"].endswith(".wav")
    assert OUTPUT_NAMES["gated_audio"] == "gated_audio.wav"


def test_manual_events_skips_detector_and_writes_events_json():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source_in = td_path / "clip.wav"
        _write_test_wav(source_in, duration=4.0)

        with patch("haptic_gt.pipeline.stitch_algorithm_output") as mock_stitch:
            mock_stitch.side_effect = lambda *args, **kwargs: None
            tracks = generate_candidate_tracks(
                source_in,
                td_path / "out",
                from_video=False,
                enable_context_detection=True,
                manual_events={
                    "category": "explosion",
                    "start_sec": 0.22,
                    "peak_sec": 0.24,
                    "end_sec": 3.66,
                },
                gate_categories=["explosion"],
            )

        assert tracks.events is not None
        assert len(tracks.events) == 1
        assert abs(tracks.events[0].peak_sec - 0.24) < 1e-6
        assert tracks.events_json is not None
        assert tracks.events_json.exists()
        payload = tracks.events_json.read_text(encoding="utf-8")
        assert "0.24" in payload
        assert '"sources": [' in payload or "manual" in payload
        assert mock_stitch.call_count == 4
