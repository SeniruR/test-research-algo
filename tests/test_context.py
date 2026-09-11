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


def test_haptic_gate_excludes_human_activity():
    tax = load_taxonomy()
    events = [
        DetectedEvent(
            category="human_activity",
            label="Chainsaw",
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
    assert len(default_gate) == 2
    assert {e.category for e in default_gate} == {"vehicle", "gunshot"}

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
    assert "vehicle" in cats
    assert "weather" in cats
    assert "human_activity" not in cats


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


def test_propose_keeps_quiet_early_and_loud_late():
    """Padded windows must not merge a quiet early blast into a louder later peak."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 4.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(1)
    early = rng.standard_normal(t.shape[0]).astype(np.float32)
    early *= 0.55 * np.exp(-((t - 0.24) ** 2) / (2 * 0.008**2))
    late = rng.standard_normal(t.shape[0]).astype(np.float32)
    late *= 1.0 * np.exp(-((t - 1.99) ** 2) / (2 * 0.03**2))
    audio = (early + late).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        windows = propose_onsets(wav_path, tax)
        centers = sorted(w.center_sec for w in windows)
        assert any(abs(c - 0.24) < 0.15 for c in centers), centers
        assert any(abs(c - 1.99) < 0.20 for c in centers), centers


def test_events_from_manual():
    from haptic_gt.context.manual_events import events_from_manual

    events = events_from_manual(
        {
            "category": "explosion",
            "start_sec": 0.22,
            "peak_sec": 0.24,
            "end_sec": 3.66,
        }
    )
    assert len(events) == 1
    assert events[0].category == "explosion"
    assert abs(events[0].peak_sec - 0.24) < 1e-6
    assert events[0].sources == ["manual"]


def test_refine_snaps_late_hint_to_early_blast_on_short_clip():
    """Short-clip lookback: AST hint near rumble must snap to earlier muzzle."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 4.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(2)
    early = rng.standard_normal(t.shape[0]).astype(np.float32)
    early *= 0.9 * np.exp(-((t - 0.24) ** 2) / (2 * 0.008**2))
    late = rng.standard_normal(t.shape[0]).astype(np.float32)
    late *= 1.1 * np.exp(-((t - 1.99) ** 2) / (2 * 0.04**2))
    audio = (early + late).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "test.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent(
                category="explosion",
                label="Explosion",
                start_sec=1.5,
                peak_sec=1.99,
                end_sec=2.5,
                confidence=0.67,
            )
        ]
        refined = refine_event_timing(events, wav_path, tax)
        assert abs(refined[0].peak_sec - 0.24) < 0.12, refined[0].peak_sec


def test_dedupe_after_snap_collapses_duplicate_peaks():
    from haptic_gt.context.frozen_fusion import dedupe_events_by_peak

    events = [
        DetectedEvent("explosion", "Explosion", 0.17, 0.25, 1.10, 0.64),
        DetectedEvent("explosion", "Explosion", 0.17, 0.25, 1.10, 0.48),
    ]
    merged = dedupe_events_by_peak(events)
    assert len(merged) == 1
    assert abs(merged[0].confidence - 0.64) < 1e-6


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


def test_refine_snaps_muzzle_on_long_clip():
    """A comparably loud muzzle just before the boom still wins the onset."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 25.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(3)
    early = rng.standard_normal(t.shape[0]).astype(np.float32)
    early *= 1.0 * np.exp(-((t - 9.90) ** 2) / (2 * 0.008**2))
    late = rng.standard_normal(t.shape[0]).astype(np.float32)
    late *= 1.15 * np.exp(-((t - 10.12) ** 2) / (2 * 0.04**2))
    audio = (early + late).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "long.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("explosion", "Explosion", 9.95, 10.12, 10.8, 0.82)
        ]
        refined = refine_event_timing(events, wav_path, tax)
        assert abs(refined[0].peak_sec - 9.90) < 0.12, refined[0].peak_sec


def test_refine_keeps_blast_and_ignores_earlier_clank():
    """A weak clank before the shot must not steal the onset."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(7)
    clank = rng.standard_normal(t.shape[0]).astype(np.float32)
    clank *= 0.18 * np.exp(-((t - 9.24) ** 2) / (2 * 0.006**2))
    shot = rng.standard_normal(t.shape[0]).astype(np.float32)
    shot *= 1.15 * np.exp(-((t - 10.12) ** 2) / (2 * 0.02**2))
    audio = (clank + shot).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "clank.wav"
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        events = [DetectedEvent("explosion", "Explosion", 10.0, 10.12, 10.8, 0.82)]
        refined = refine_event_timing(events, wav_path, tax)
        assert abs(refined[0].peak_sec - 10.12) < 0.12, refined[0].peak_sec


def test_refine_does_not_drag_volley_shot_backwards():
    """Each shot in a volley keeps its own onset (no 0.9 s backward slide)."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(14)
    audio = np.zeros_like(t)
    for peak_t, amp in ((11.00, 0.62), (11.52, 1.0), (13.40, 0.86)):
        burst = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + burst * (amp * np.exp(-((t - peak_t) ** 2) / (2 * 0.012**2)))
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "volley.wav"
        sf.write(wav_path, audio.astype(np.float32), sr, subtype="PCM_16")
        events = [
            DetectedEvent("explosion", "Explosion", 11.4, 11.52, 12.0, 0.75),
            DetectedEvent("explosion", "Explosion", 13.3, 13.40, 13.9, 0.75),
        ]
        refined = refine_event_timing(events, wav_path, tax)
        peaks = sorted(e.peak_sec for e in refined)
        assert abs(peaks[0] - 11.52) < 0.12, peaks
        assert abs(peaks[1] - 13.40) < 0.12, peaks


def test_promote_recovers_cannon_labeled_vehicle():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    sr = 44100
    duration = 4.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(4)
    blast = rng.standard_normal(t.shape[0]).astype(np.float32)
    blast *= 0.9 * np.exp(-((t - 1.5) ** 2) / (2 * 0.008**2))
    bed = (0.08 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    audio = (bed + blast).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "cannon.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [DetectedEvent("vehicle", "Vehicle", 1.0, 1.5, 2.2, 0.66)]
        scores = [
            EncoderScore(1.5, "Explosion", 0.22, "audio"),
            EncoderScore(1.5, "Vehicle", 0.66, "audio"),
        ]
        out = promote_impulsive_transients(events, wav, scores, tax)
        explosions = [e for e in out if e.category == "explosion"]
        assert len(explosions) >= 1
        assert abs(explosions[0].peak_sec - 1.5) < 0.2


def test_promote_recovers_quieter_volley_shots():
    """Later shots quieter than the first cannon must still be promoted."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(8)
    audio = (0.08 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    shots = [9.64, 11.16, 11.66, 12.45, 13.50, 14.55]
    amps = [1.0, 0.75, 0.72, 0.70, 0.68, 0.65]
    for peak_t, amp in zip(shots, amps):
        burst = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + burst * (amp * np.exp(-((t - peak_t) ** 2) / (2 * 0.008**2)))
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "volley.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("explosion", "Explosion", 9.56, 9.64, 10.4, 0.82),
            DetectedEvent("vehicle", "Vehicle", 14.9, 15.4, 15.8, 0.66),
        ]
        scores = [
            EncoderScore(9.64, "Explosion", 0.82, "audio"),
            EncoderScore(11.16, "Explosion", 0.42, "audio"),
            EncoderScore(11.66, "Explosion", 0.40, "audio"),
            EncoderScore(12.45, "Explosion", 0.38, "audio"),
            EncoderScore(13.50, "Vehicle", 0.50, "audio"),
            EncoderScore(14.55, "Vehicle", 0.48, "audio"),
        ]
        out = promote_impulsive_transients(events, wav, scores, tax)
        peaks = sorted(e.peak_sec for e in out if e.category == "explosion")
        assert len(peaks) >= 5, peaks
        # 13.50 has no explosion score: recovered by shot-level flux alone
        for want in (9.64, 11.16, 11.66, 12.45, 13.50):
            assert any(abs(p - want) < 0.25 for p in peaks), (want, peaks)


def test_promote_ignores_track_clanks_in_drive():
    """Sharp but quiet tank-track clanks must not become explosions."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(11)
    audio = (0.10 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    clank_times = np.arange(0.4, 9.0, 0.8)
    for peak_t in clank_times:
        clank = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + clank * (0.10 * np.exp(-((t - peak_t) ** 2) / (2 * 0.006**2)))
    for peak_t in (9.64, 11.16):
        shot = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + shot * (1.0 * np.exp(-((t - peak_t) ** 2) / (2 * 0.010**2)))
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "clanks.wav"
        sf.write(wav, audio.astype(np.float32), sr, subtype="PCM_16")
        events = [DetectedEvent("explosion", "Explosion", 9.56, 9.64, 10.4, 0.82)]
        scores = [
            EncoderScore(9.64, "Explosion", 0.82, "audio"),
            EncoderScore(11.16, "Explosion", 0.75, "audio"),
            EncoderScore(2.0, "Explosion", 0.19, "audio"),
            EncoderScore(5.0, "Explosion", 0.18, "audio"),
        ]
        out = promote_impulsive_transients(events, wav, scores, tax)
        explosions = [e for e in out if e.category == "explosion"]
        assert all(e.peak_sec > 9.0 for e in explosions), [
            round(e.peak_sec, 2) for e in explosions
        ]
        assert any(abs(e.peak_sec - 11.16) < 0.25 for e in explosions)


def test_demote_rumble_false_explosion():
    """Steady tank drive must not stay labeled as an explosion."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    sr = 44100
    duration = 12.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(9)
    bed = (0.28 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    blast = rng.standard_normal(t.shape[0]).astype(np.float32)
    blast *= 0.95 * np.exp(-((t - 8.0) ** 2) / (2 * 0.008**2))
    audio = (bed + blast).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "drive.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("explosion", "Explosion", 1.8, 1.94, 2.8, 0.82),
            DetectedEvent("explosion", "Explosion", 7.9, 8.0, 8.7, 0.82),
        ]
        scores = [
            EncoderScore(1.94, "Explosion", 0.82, "audio"),
            EncoderScore(8.0, "Explosion", 0.82, "audio"),
        ]
        out = promote_impulsive_transients(events, wav, scores, tax)
        explosions = [e for e in out if e.category == "explosion"]
        assert all(abs(e.peak_sec - 1.94) > 0.5 for e in explosions), [
            e.peak_sec for e in explosions
        ]
        assert any(abs(e.peak_sec - 8.0) < 0.2 for e in explosions)


def test_clip_start_is_not_a_transient():
    """Silence-to-signal at t=0 must not look like an attack."""
    from haptic_gt.context.onset_refine import _spectral_flux, local_flux_ratio

    sr = 44100
    duration = 3.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(12)
    audio = (0.2 * rng.standard_normal(t.shape[0])).astype(np.float32)
    times, flux = _spectral_flux(audio, sr, hop_ms=5.0)
    assert local_flux_ratio(times, flux, float(times[1])) == 0.0
    assert local_flux_ratio(times, flux, 1.5) > 0.0


def test_shot_calibration_marks_hit_miss_and_fp():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.shot_calib import (
        calibrate_shot_times,
        format_shot_calibration_table,
        scan_shot_attacks,
    )

    sr = 44100
    duration = 8.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(13)
    audio = (0.05 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    for peak_t in (2.0, 5.0):
        shot = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + shot * (1.0 * np.exp(-((t - peak_t) ** 2) / (2 * 0.01**2)))
    events = {
        "events": [
            {
                "event_id": "event_001",
                "category": "explosion",
                "peak_sec": 2.02,
                "confidence": 0.81,
                "audio_score": 0.81,
            },
            {
                "event_id": "event_002",
                "category": "explosion",
                "peak_sec": 7.0,
                "confidence": 0.62,
                "audio_score": 0.62,
            },
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, audio.astype(np.float32), sr, subtype="PCM_16")
        report = calibrate_shot_times(wav, [2.0, 5.0], events)
        scan = scan_shot_attacks(wav, top_n=10)

    assert report["summary"]["hits"] == 1
    assert report["summary"]["misses"] == 1
    assert report["summary"]["false_positives"] == 1
    missed = [r for r in report["manual_shots"] if r["status"] == "miss"][0]
    # The 5.0 s shot is real audio even though no event matched it
    assert abs(missed["nearest_attack_sec"] - 5.0) < 0.05
    assert missed["flux_rel_shot_level"] > 0.5
    table = format_shot_calibration_table(report)
    assert "event_001" in table
    assert any(abs(a["t_sec"] - 5.0) < 0.05 for a in scan["attacks"])


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


def _shot_clip(sr: int = 22050, duration: float = 4.0, shots=((1.0, 1.0), (2.5, 0.8))):
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(3)
    audio = (0.08 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    audio = audio + rng.standard_normal(t.size).astype(np.float32) * 0.05
    for peak_t, amp in shots:
        audio = audio + rng.standard_normal(t.size).astype(np.float32) * (
            amp * np.exp(-((t - peak_t) ** 2) / (2 * 0.010**2))
        )
    return audio.astype(np.float32)


def test_accents_closer_than_min_distance_collapse_to_the_stronger():
    """One blast reported twice is one bang: the weaker report goes."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_nms import suppress_impulsive_overlaps

    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, _shot_clip(), 22050, subtype="PCM_16")
        # 0.39 s apart, inside impulsive_min_peak_distance_sec
        out = suppress_impulsive_overlaps(
            [
                DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.85, 0.30),
                DetectedEvent("explosion", "Explosion", 1.31, 1.39, 1.90, 0.30),
            ],
            wav,
            tax,
        )

    peaks = sorted(e.peak_sec for e in out)
    assert len(peaks) == 1, peaks
    assert abs(peaks[0] - 1.00) < 1e-6, peaks


def test_accent_spans_are_trimmed_so_they_do_not_overlap():
    """A decay tail must end before the next accent starts, or one bang blurs."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_nms import suppress_impulsive_overlaps

    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, _shot_clip(), 22050, subtype="PCM_16")
        out = suppress_impulsive_overlaps(
            [
                DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.95, 0.30),
                DetectedEvent("explosion", "Explosion", 2.42, 2.50, 3.20, 0.30),
            ],
            wav,
            tax,
        )

    shots = sorted(out, key=lambda e: e.peak_sec)
    assert len(shots) == 2, shots
    assert shots[0].end_sec <= shots[1].start_sec + 1e-6, (shots[0], shots[1])
    assert shots[0].end_sec > shots[0].peak_sec


def test_rumble_bed_survives_a_shot_fired_inside_it():
    """Vibration must not cut out the moment the cannon fires.

    A short sustained chip sitting on a blast is that blast mislabelled and is
    dropped; a long engine bed that merely happens to peak under one shot is the
    engine, and it keeps running.
    """
    import tempfile
    from pathlib import Path

    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    scores = [
        EncoderScore(time_sec=float(t), label="Explosion", score=0.5, source="audio")
        for t in np.arange(0.4, 3.6, 0.2)
    ]
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, _shot_clip(), 22050, subtype="PCM_16")
        out = promote_impulsive_transients(
            [DetectedEvent("vehicle", "Vehicle", 0.05, 1.02, 3.95, 0.6)],
            wav,
            scores,
            tax,
        )

    vehicle = [e for e in out if e.category == "vehicle"]
    assert vehicle, [e.category for e in out]
    assert vehicle[0].end_sec - vehicle[0].start_sec > 3.0
    assert any(e.category in ("explosion", "gunshot") for e in out), out


def test_refine_keeps_peaks_when_relocation_is_off():
    """Rebuilding spans must not walk a peak off the attack it is already on."""
    import tempfile
    from pathlib import Path

    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, _shot_clip(), 22050, subtype="PCM_16")
        events = [DetectedEvent("explosion", "Explosion", 2.42, 2.50, 2.95, 0.30)]
        kept = refine_event_timing(events, wav, tax, relocate_impulsive_peaks=False)
        moved = refine_event_timing(events, wav, tax)

    assert kept[0].peak_sec == 2.50
    # Relocation would otherwise reach back to the louder shot at 1.0 s
    assert moved[0].peak_sec < 2.50
    assert kept[0].end_sec > kept[0].peak_sec


def test_events_json_reports_attack_strength():
    """Flat posteriors hide which bang was loud; the attack numbers must show it."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.detector import EventResult
    from haptic_gt.context.impulsive_nms import measure_impulsive_attacks

    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "shots.wav"
        sf.write(wav, _shot_clip(), 22050, subtype="PCM_16")
        events = measure_impulsive_attacks(
            [
                DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.45, 0.30),
                DetectedEvent("explosion", "Explosion", 2.42, 2.50, 2.95, 0.30),
                DetectedEvent("vehicle", "Vehicle", 0.05, 2.00, 3.95, 0.60),
            ],
            wav,
            tax,
        )

    rows = EventResult(events=events).to_dict()["events"]
    shots = [r for r in rows if r["category"] == "explosion"]
    assert all(r["confidence"] == 0.30 for r in shots)
    # Same confidence, and the loudest bang still reads as the loudest
    assert shots[0]["attack_rel_max"] > shots[1]["attack_rel_max"], shots
    assert all(r["attack_prominence"] > 2.0 for r in shots), shots
    assert "attack_rel_max" not in [r for r in rows if r["category"] == "vehicle"][0]


if __name__ == "__main__":
    test_taxonomy_mapping()
    test_event_mask()
    test_haptic_gate_excludes_human_activity()
    test_gate_categories_override()
    test_multi_event_mask()
    test_resolve_gate_categories_default()
    test_propose_onsets_finds_synthetic_peak()
    test_propose_min_distance()
    test_propose_keeps_quiet_early_and_loud_late()
    test_events_from_manual()
    test_refine_snaps_late_hint_to_early_blast_on_short_clip()
    test_dedupe_after_snap_collapses_duplicate_peaks()
    test_fusion_weather()
    test_fusion_gunshot_audio_only()
    test_peak_split_multiple_blasts()
    test_refine_impulsive_peak_forward()
    test_haptic_onset_alignment()
    test_propose_all_includes_sustained_scan()
    test_refine_impulsive_peak()
    test_debug_events_table()
    print("All context unit tests passed.")
