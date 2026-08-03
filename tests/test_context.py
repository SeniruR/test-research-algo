"""Tests for context detection (no heavy model downloads)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from haptic_gt.context.event_aggregation import aggregate_events
from haptic_gt.context.context_detectors import SymbolicToken
from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.frozen_fusion import fuse_events
from haptic_gt.context.mask import (
    build_event_mask,
    events_for_haptic_gate,
    resolve_gate_categories,
)
from haptic_gt.context.onset_refine import refine_event_timing
from haptic_gt.context.proposals import propose_onsets
from haptic_gt.context.taxonomy import load_taxonomy, match_label_to_category
from haptic_gt.context.frozen_fusion import DetectedEvent


def test_taxonomy_mapping():
    tax = load_taxonomy()
    assert match_label_to_category(tax, "Thunder", "audio") == "weather"
    assert match_label_to_category(tax, "Gunshot, gunfire", "audio") == "gunshot"
    assert match_label_to_category(tax, "Artillery fire", "audio") == "explosion"
    assert match_label_to_category(tax, "chopping wood", "video") == "human_activity"


def test_event_mask():
    sr = 44100
    events = [
        DetectedEvent(
            category="weather",
            label="thunder",
            start_sec=1.0,
            peak_sec=1.5,
            end_sec=2.0,
            confidence=0.9,
        )
    ]
    mask = build_event_mask(sr * 3, sr, events)
    assert mask.shape == (sr * 3,)
    assert mask[int(1.5 * sr)] > 0.5
    assert mask[0] == 0.0


def test_haptic_gate_excludes_vehicle():
    tax = load_taxonomy()
    events = [
        DetectedEvent(
            category="vehicle",
            label="Vehicle",
            start_sec=0.0,
            peak_sec=5.0,
            end_sec=11.0,
            confidence=0.6,
        ),
        DetectedEvent(
            category="explosion",
            label="Explosion",
            start_sec=13.5,
            peak_sec=14.0,
            end_sec=15.0,
            confidence=0.7,
        ),
    ]
    gated = events_for_haptic_gate(events, tax)
    assert len(gated) == 1
    assert gated[0].category == "explosion"


def test_gate_categories_override():
    tax = load_taxonomy()
    events = [
        DetectedEvent(
            category="vehicle",
            label="Vehicle",
            start_sec=1.0,
            peak_sec=2.0,
            end_sec=3.0,
            confidence=0.6,
        ),
        DetectedEvent(
            category="gunshot",
            label="Gunshot, gunfire",
            start_sec=5.0,
            peak_sec=5.5,
            end_sec=6.0,
            confidence=0.8,
        ),
    ]
    default_gate = events_for_haptic_gate(events, tax)
    assert len(default_gate) == 1
    assert default_gate[0].category == "gunshot"

    custom_gate = events_for_haptic_gate(
        events, tax, gate_categories=["vehicle", "human_activity"]
    )
    assert len(custom_gate) == 1
    assert custom_gate[0].category == "vehicle"


def test_multi_event_mask():
    sr = 44100
    events = [
        DetectedEvent("gunshot", "Gunshot", 1.0, 1.2, 1.5, 0.9),
        DetectedEvent("gunshot", "Gunshot", 3.0, 3.2, 3.5, 0.9),
        DetectedEvent("gunshot", "Gunshot", 5.0, 5.2, 5.5, 0.9),
        DetectedEvent("vehicle", "Vehicle", 10.0, 12.0, 14.0, 0.7),
        DetectedEvent("vehicle", "Vehicle", 20.0, 22.0, 24.0, 0.7),
        DetectedEvent("vehicle", "Vehicle", 30.0, 32.0, 34.0, 0.7),
        DetectedEvent("vehicle", "Vehicle", 40.0, 42.0, 44.0, 0.7),
    ]
    mask = build_event_mask(sr * 50, sr, events)
    assert mask[int(1.2 * sr)] > 0.5
    assert mask[int(12 * sr)] > 0.5
    assert mask[int(0.5 * sr)] == 0.0


def test_resolve_gate_categories_default():
    tax = load_taxonomy()
    cats = resolve_gate_categories(tax)
    assert "gunshot" in cats
    assert "explosion" in cats
    assert "vehicle" not in cats


def test_propose_onsets_finds_synthetic_peak():
    import tempfile
    from pathlib import Path

    sr = 44100
    duration = 5.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = np.exp(-((t - 2.0) ** 2) / (2 * 0.03**2)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        windows = propose_onsets(wav_path)
        assert len(windows) >= 1
        best = max(windows, key=lambda w: w.rms_score)
        assert abs(best.center_sec - 2.0) < 0.2


def test_propose_min_distance():
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 10.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    for peak_t in (2.0, 2.3, 5.0):
        audio += np.exp(-((t - peak_t) ** 2) / (2 * 0.02**2))

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        windows = propose_onsets(wav_path, tax)
        centers = sorted(w.center_sec for w in windows)
        assert len(centers) >= 2
        if len(centers) >= 2:
            assert centers[1] - centers[0] >= tax.impulsive_min_peak_distance_sec * 0.8


def test_fusion_weather():
    tax = load_taxonomy()
    tokens = [
        SymbolicToken(
            time_sec=2.0,
            label="Thunder",
            confidence=0.9,
            modality="audio",
            category="weather",
        )
    ]
    scores = [
        EncoderScore(time_sec=2.0, label="Thunder", score=0.7, source="audio"),
    ]
    events = fuse_events(tokens, scores, tax)
    assert len(events) >= 1
    assert events[0].category == "weather"


def test_fusion_gunshot_audio_only():
    tax = load_taxonomy()
    scores = [
        EncoderScore(time_sec=14.0, label="Explosion", score=0.55, source="audio"),
        EncoderScore(time_sec=14.5, label="Explosion", score=0.48, source="audio"),
    ]
    events = fuse_events([], scores, tax)
    assert any(e.category == "explosion" for e in events), events


def test_peak_split_multiple_blasts():
    tax = load_taxonomy()
    scores = [
        EncoderScore(time_sec=10.0, label="Explosion", score=0.72, source="audio"),
        EncoderScore(time_sec=10.25, label="Explosion", score=0.61, source="audio"),
        EncoderScore(time_sec=12.0, label="Gunshot, gunfire", score=0.68, source="audio"),
        EncoderScore(time_sec=12.25, label="Gunshot, gunfire", score=0.55, source="audio"),
        EncoderScore(time_sec=14.5, label="Artillery fire", score=0.78, source="audio"),
        EncoderScore(time_sec=14.75, label="Artillery fire", score=0.66, source="audio"),
        EncoderScore(time_sec=16.5, label="Explosion", score=0.70, source="audio"),
        EncoderScore(time_sec=5.0, label="Vehicle", score=0.60, source="audio"),
        EncoderScore(time_sec=5.5, label="Vehicle", score=0.58, source="audio"),
        EncoderScore(time_sec=6.0, label="Vehicle", score=0.55, source="audio"),
    ]
    events = fuse_events([], scores, tax)
    impulsive = [e for e in events if e.category in ("gunshot", "explosion")]
    assert len(impulsive) >= 3, events
    assert any(e.category == "vehicle" for e in events), events
    for e in impulsive:
        assert e.end_sec - e.start_sec <= 1.5, e


def test_refine_impulsive_peak_forward():
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 6.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    # Soft precursor then sharp broadband blast (spectral flux should pick 2.0s)
    audio = 0.25 * np.sin(2 * np.pi * 120 * t) * np.exp(-((t - 1.5) ** 2) / (2 * 0.04**2))
    blast = (np.random.default_rng(0).standard_normal(t.shape[0]).astype(np.float32))
    blast *= np.exp(-((t - 2.0) ** 2) / (2 * 0.012**2))
    audio = (audio + blast).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")

        events = [
            DetectedEvent(
                category="explosion",
                label="Explosion",
                start_sec=1.0,
                peak_sec=1.55,
                end_sec=2.0,
                confidence=0.7,
            )
        ]
        refined = refine_event_timing(events, wav_path, tax)
        assert abs(refined[0].peak_sec - 2.0) < 0.08
        assert refined[0].start_sec <= refined[0].peak_sec
        assert refined[0].end_sec > refined[0].peak_sec


def test_haptic_onset_alignment():
    import tempfile
    from pathlib import Path

    from haptic_gt.haptic_synthesis import stitch_algorithm_output

    sr = 44100
    duration = 4.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    audio += np.exp(-((t - 2.0) ** 2) / (2 * 0.01**2))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        out = td_path / "haptic.wav"
        sf.write(source, audio.astype(np.float32), sr, subtype="PCM_16")

        events = [
            DetectedEvent("explosion", "Explosion", 1.9, 2.0, 2.4, 0.9),
        ]

        def _fake_algo(inp, outp):
            # Fake haptic: silence then attack, mimicking algorithm delay
            n = int(0.8 * 8000)
            seg = np.zeros(n, dtype=np.float32)
            attack = int(0.25 * 8000)
            seg[attack:] = 0.5
            sf.write(outp, seg, 8000, subtype="PCM_16")

        stitch_algorithm_output(source, events, out, _fake_algo)
        haptic, out_sr = sf.read(out)
        assert out_sr == 8000
        # Haptic attack (~0.25s into segment) should land near peak_sec=2.0
        thr = 0.1 * float(np.max(np.abs(haptic)))
        onset_idx = int(np.where(np.abs(haptic) >= thr)[0][0])
        onset_sec = onset_idx / out_sr
        assert abs(onset_sec - 2.0) < 0.05


def test_propose_all_includes_sustained_scan():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.proposals import propose_all_windows

    sr = 44100
    duration = 10.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = (0.05 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        windows = propose_all_windows(wav_path)
        assert len(windows) >= 4


def test_refine_impulsive_peak():
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 5.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = np.exp(-((t - 2.0) ** 2) / (2 * 0.03**2)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")

        events = [
            DetectedEvent(
                category="explosion",
                label="Explosion",
                start_sec=1.5,
                peak_sec=2.35,
                end_sec=2.7,
                confidence=0.7,
            )
        ]
        refined = refine_event_timing(events, wav_path, tax)
        assert abs(refined[0].peak_sec - 2.0) < 0.15
        assert refined[0].end_sec > refined[0].peak_sec


def test_debug_events_table():
    from haptic_gt.context.debug_events import EventDebugRow, format_events_table

    rows = [
        EventDebugRow(
            event_id="event_001",
            category="explosion",
            label="Explosion",
            start_sec=14.9,
            peak_sec=15.0,
            end_sec=15.8,
            duration_sec=0.9,
            confidence=0.82,
            audio_score=0.82,
            video_score=None,
            sources=["audio"],
            included_in_gate=True,
        )
    ]
    table = format_events_table(rows)
    assert "event_001" in table
    assert "15.000" in table


if __name__ == "__main__":
    test_taxonomy_mapping()
    test_event_mask()
    test_haptic_gate_excludes_vehicle()
    test_gate_categories_override()
    test_multi_event_mask()
    test_resolve_gate_categories_default()
    test_propose_onsets_finds_synthetic_peak()
    test_propose_min_distance()
    test_fusion_weather()
    test_fusion_gunshot_audio_only()
    test_peak_split_multiple_blasts()
    test_refine_impulsive_peak_forward()
    test_haptic_onset_alignment()
    test_propose_all_includes_sustained_scan()
    test_refine_impulsive_peak()
    test_debug_events_table()
    print("All context unit tests passed.")
