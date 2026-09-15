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


def _rms(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> float:
    seg = audio[int(start_sec * sr) : int(end_sec * sr)]
    return float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0


def _combined_scenes_audio(sr: int = 22050):
    """Three clips cut together, as the real test video is.

    Cannon volley over near silence, then a tank drive recorded far away, then a
    car rumble recorded close -- four times the drive's level -- over its own idle
    bed. The car must not set the loudness floor for the drive.
    """
    shots = (1.5, 2.45, 2.95, 3.7, 4.85, 6.45, 7.8)
    drive = (10.0, 18.0)
    bursts = ((18.8, 20.8), (21.7, 23.2), (24.0, 25.4), (26.2, 27.8))

    rng = np.random.default_rng(3)
    n = int(29.1 * sr)
    t = np.arange(n) / sr
    audio = (0.004 * rng.standard_normal(n)).astype(np.float32)

    for s in shots:
        i0 = int(s * sr)
        tt = np.arange(n - i0) / sr
        body = np.exp(-tt * 7.0) * np.sin(2 * np.pi * 65 * tt)
        crack = np.exp(-tt * 240.0) * rng.standard_normal(tt.size)
        audio[i0:] += (0.80 * body + 0.45 * crack).astype(np.float32)

    in_drive = ((t >= drive[0]) & (t <= drive[1])).astype(np.float32)
    audio += (0.22 * np.sin(2 * np.pi * 42 * t) * in_drive).astype(np.float32)
    for clank in np.arange(drive[0] + 0.5, drive[1], 0.9):
        i0 = int(clank * sr)
        tt = np.arange(min(int(0.15 * sr), n - i0)) / sr
        audio[i0 : i0 + tt.size] += (
            0.18 * np.exp(-tt * 60.0) * rng.standard_normal(tt.size)
        ).astype(np.float32)

    in_car = ((t >= 18.4) & (t <= 29.0)).astype(np.float32)
    audio += (0.30 * np.sin(2 * np.pi * 48 * t) * in_car).astype(np.float32)
    for a, b in bursts:
        m = ((t >= a) & (t <= b)).astype(np.float32)
        audio += (0.55 * np.sin(2 * np.pi * 48 * t) * m).astype(np.float32)

    # Engines are not sine waves. Without a broadband part the mid-band the
    # frequency-shifting algorithm listens to is empty and its rumble comes out
    # of the filter noise instead of the engine.
    noise = rng.standard_normal(n).astype(np.float32)
    audio += 0.05 * noise * in_drive
    audio += 0.09 * noise * in_car

    return np.clip(audio, -1.0, 1.0), shots, drive, bursts


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


def test_vehicle_uses_bed_not_bang_accent():
    """Sparse vehicle detection keeps full bed (no bang accent at fake peak)."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        # Steady tone = rumble stand-in; no real transient
        _write_test_wav(source, duration=4.0)

        events = [
            DetectedEvent("vehicle", "Vehicle", 0.16, 0.725, 1.43, 0.46),
        ]

        def _passthrough(in_wav, out_wav):
            audio, sr = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, audio, sr, subtype="PCM_16")

        profile = ContinuousProfile()
        with_vehicle = td_path / "with_vehicle.wav"
        bed_only = td_path / "bed_only.wav"
        stitch_algorithm_output(
            source, events, with_vehicle, _passthrough, continuous=profile
        )
        stitch_algorithm_output(
            source, [], bed_only, _passthrough, continuous=profile
        )

        a, sr = sf.read(with_vehicle)
        b, _ = sf.read(bed_only)
        # Same continuous bed; no extra bang accent energy at the fake peak
        peak_slice = slice(int(0.5 * sr), int(1.0 * sr))
        assert np.max(np.abs(a[peak_slice] - b[peak_slice])) < 0.05
        assert np.corrcoef(a, b)[0, 1] > 0.99


def test_merge_sustained_collapses_vehicle_chips():
    from haptic_gt.context.sustained_merge import merge_sustained_events

    chips = [
        DetectedEvent("vehicle", "Vehicle", 6.5, 6.7, 7.0, 0.4),
        DetectedEvent("vehicle", "Vehicle", 7.2, 7.2, 7.7, 0.3),  # 0.2s gap → merge
        DetectedEvent("vehicle", "Vehicle", 8.8, 8.8, 9.3, 0.35),  # 1.1s gap → keep
        DetectedEvent("explosion", "Explosion", 10.0, 10.1, 10.5, 0.8),
    ]
    merged = merge_sustained_events(chips)
    vehicles = [e for e in merged if e.category == "vehicle"]
    explosions = [e for e in merged if e.category == "explosion"]
    assert len(explosions) == 1
    assert len(vehicles) == 2
    assert abs(vehicles[0].start_sec - 6.5) < 1e-6
    assert vehicles[0].end_sec >= 7.7
    assert abs(vehicles[1].start_sec - 8.8) < 1e-6


def test_merge_does_not_bridge_far_rumble_bursts():
    from haptic_gt.context.sustained_merge import merge_sustained_events

    bursts = [
        DetectedEvent("vehicle", "Vehicle", 34.0, 34.5, 35.0, 0.4),
        DetectedEvent("vehicle", "Vehicle", 37.0, 37.5, 38.0, 0.4),
        DetectedEvent("vehicle", "Vehicle", 41.0, 41.5, 42.0, 0.4),
    ]
    merged = merge_sustained_events(bursts)
    assert len(merged) == 3


def test_rumble_filter_drops_steady_bed_keeps_burst():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 8.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.12 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    gate = ((t >= 4.7) & (t <= 5.4)).astype(np.float32)
    burst = (0.55 * np.sin(2 * np.pi * 55 * t) * gate).astype(np.float32)
    audio = bed + burst

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "eng.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("vehicle", "Vehicle", 1.5, 2.0, 2.5, 0.25),  # flat + weak
            DetectedEvent("vehicle", "Vehicle", 4.7, 5.0, 5.4, 0.40),  # real burst
            DetectedEvent("explosion", "Explosion", 6.0, 6.1, 6.5, 0.8),
        ]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        cats = [e.category for e in filtered]
        assert "explosion" in cats
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert any(v.start_sec <= 4.9 and v.end_sec >= 5.2 for v in vehicles)
        assert not any(abs(v.peak_sec - 2.0) < 0.4 for v in vehicles)


def test_rumble_filter_drops_sub_half_second_bed_ticks():
    """Engine-bed AST chips (~0.2 s) must not become rumble events."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 10.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.08 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    spike = (
        0.55 * np.sin(2 * np.pi * 55 * t) * np.exp(-((t - 3.0) ** 2) / (2 * 0.04**2))
    ).astype(np.float32)
    plateau = (
        0.55 * np.sin(2 * np.pi * 55 * t) * ((t >= 6.0) & (t <= 7.2)).astype(np.float32)
    ).astype(np.float32)
    audio = bed + spike + plateau
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "ticks.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("vehicle", "Vehicle", 2.9, 3.0, 3.1, 0.4),
            DetectedEvent("vehicle", "Vehicle", 5.8, 6.5, 7.3, 0.4),
        ]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert any(e.start_sec <= 6.2 and e.end_sec >= 7.0 for e in vehicles)
        assert not any(abs(e.peak_sec - 3.0) < 0.25 for e in vehicles)


def test_rumble_filter_fills_loud_plateau_without_ast():
    """AST-missed loud rumble still becomes a vehicle span."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 12.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.08 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    gate = ((t >= 3.0) & (t <= 5.5)).astype(np.float32)
    plateau = (0.55 * np.sin(2 * np.pi * 55 * t) * gate).astype(np.float32)
    bump = (
        0.16 * np.sin(2 * np.pi * 50 * t) * np.exp(-((t - 8.0) ** 2) / (2 * 0.08**2))
    ).astype(np.float32)
    audio = bed + plateau + bump

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "eng.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("vehicle", "Vehicle", 1.5, 2.0, 2.5, 0.35),
            DetectedEvent("vehicle", "Vehicle", 7.7, 8.0, 8.3, 0.32),
            DetectedEvent("explosion", "Explosion", 10.0, 10.1, 10.4, 0.8),
        ]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert any(e.start_sec <= 3.2 and e.end_sec >= 5.2 for e in vehicles)
        assert not any(abs(e.peak_sec - 8.0) < 0.35 for e in vehicles)
        assert any(e.category == "explosion" for e in filtered)


def test_rumble_filter_keeps_drive_when_cannons_present():
    """0–N tank drive must still island after later cannons."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    drive = (0.30 * np.sin(2 * np.pi * 40 * t) * ((t >= 0.2) & (t <= 7.5)).astype(np.float32))
    blast = (
        0.95 * np.sin(2 * np.pi * 200 * t) * np.exp(-((t - 9.64) ** 2) / (2 * 0.01**2))
    ).astype(np.float32)
    late = (
        0.40 * np.sin(2 * np.pi * 55 * t) * ((t >= 14.8) & (t <= 15.7)).astype(np.float32)
    )
    audio = (drive + blast + late).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mixed.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("explosion", "Explosion", 9.5, 9.64, 10.3, 0.82),
            DetectedEvent("vehicle", "Vehicle", 14.9, 15.4, 15.8, 0.66),
        ]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert any(e.start_sec <= 1.5 and e.end_sec >= 3.0 for e in vehicles), [
            (e.start_sec, e.end_sec) for e in vehicles
        ]


def test_salience_threshold_ignores_cannon_outliers():
    """Explosions must not set the vehicle loudness floor above tank drive."""
    from haptic_gt.context.onset_refine import _envelope_rms
    from haptic_gt.context.sustained_salience import salience_stats
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 8.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.25 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    blast = (
        0.9 * np.sin(2 * np.pi * 200 * t) * ((t >= 5.5) & (t <= 6.5)).astype(np.float32)
    ).astype(np.float32)
    times, env = _envelope_rms(bed + blast, sr, hop_ms=5.0)
    raw = salience_stats(env, tax)
    masked = salience_stats(
        env, tax, times=times, exclude_peaks=[6.0], exclude_radius_sec=1.2
    )
    assert masked["threshold"] < raw["threshold"]
    assert masked["threshold"] < 0.45


def test_silence_does_not_collapse_salience_threshold():
    """A clip opening on digital silence must still gate on the loud bursts.

    Silence is a mode of its own; if it is left in the statistics the split lands
    just above zero and the whole clip becomes one island.
    """
    from haptic_gt.context.onset_refine import _envelope_rms
    from haptic_gt.context.sustained_salience import rms_islands, salience_stats
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 20.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = 0.30 * np.sin(2 * np.pi * 55 * t) * (t >= 5.0)
    louder = 0.60 * np.sin(2 * np.pi * 55 * t) * (
        ((t >= 8.0) & (t <= 9.0)) | ((t >= 15.0) & (t <= 16.0))
    )
    times, env = _envelope_rms((bed + louder).astype(np.float32), sr, hop_ms=5.0)

    stats = salience_stats(env, tax)
    thr = float(stats["threshold"])
    assert thr > 0.05, stats
    assert 0.20 < thr < 0.60, stats

    islands = rms_islands(
        times,
        env,
        thr,
        min_sec=tax.sustained_salience_min_sec,
        gap_sec=tax.sustained_salience_gap_sec,
        min_duty=tax.sustained_salience_min_duty,
    )
    assert len(islands) == 2, islands
    assert all(end - start < 3.0 for start, _, end in islands), islands


def test_repeated_lulls_are_a_rhythm_but_one_long_lull_is_a_camera_angle():
    """Both look like one loudness mode; only one of them has gaps to keep quiet.

    Once bursts fill more than half a scene its median sits inside them, so the
    percentiles alone cannot tell an intermittent car rumble from a drive filmed
    from a wider angle for a few seconds.
    """
    from haptic_gt.context.sustained_salience import _region_threshold
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    dt = 0.005
    n = int(12.0 / dt)
    t = np.arange(n) * dt

    rhythm = np.full(n, 0.20)
    for a, b in ((0.5, 2.0), (2.9, 4.4), (5.2, 6.8), (7.6, 9.2), (10.0, 11.8)):
        rhythm[(t >= a) & (t <= b)] = 0.60
    one_lull = np.full(n, 0.60)
    one_lull[(t >= 4.0) & (t <= 8.0)] = 0.20

    assert _region_threshold(rhythm, tax, dt=dt)["mode"] == "split"
    assert _region_threshold(one_lull, tax, dt=dt)["mode"] == "keep"
    # And the quieter angle stays above its own floor, so the drive is continuous
    assert _region_threshold(one_lull, tax, dt=dt)["threshold"] < 0.20


def test_clanks_over_a_steady_drive_are_not_a_rhythm():
    """Track clanks are loud and regular; the drive must not be chopped between them."""
    from haptic_gt.context.sustained_salience import _region_threshold
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    dt = 0.005
    n = int(9.0 / dt)
    t = np.arange(n) * dt
    series = np.full(n, 0.25)
    for clank in np.arange(0.6, 9.0, 0.7):
        series[(t >= clank) & (t <= clank + 0.06)] = 0.75

    out = _region_threshold(series, tax, dt=dt)
    assert out["mode"] == "keep", out
    assert out["threshold"] < 0.25, out


def test_island_start_follows_the_attack_ramp_not_the_gate_crossing():
    """A rumble that ramps up must buzz from the ramp, not a third of a second in."""
    from haptic_gt.context.sustained_salience import rms_islands

    dt = 0.005
    times = np.arange(0.0, 4.0, dt)
    env = np.full_like(times, 0.2)
    ramp = (times >= 1.2) & (times < 1.5)
    env[ramp] = 0.2 + 0.8 * (times[ramp] - 1.2) / 0.3
    env[(times >= 1.5) & (times <= 2.5)] = 1.0

    islands = rms_islands(times, env, 0.9, min_sec=0.5, gap_sec=0.18)
    assert len(islands) == 1, islands
    start, _, _ = islands[0]
    assert 1.2 < start < 1.45, start


def test_island_edges_never_produce_overlapping_bursts():
    """Both edges walking into the same lull must yield one burst, not two."""
    from haptic_gt.context.sustained_salience import rms_islands

    dt = 0.005
    times = np.arange(0.0, 5.0, dt)
    env = np.full_like(times, 0.2)
    env[(times >= 1.0) & (times <= 2.0)] = 1.0
    env[(times > 2.0) & (times < 2.5)] = 0.8  # above the edge floor, below the gate
    env[(times >= 2.5) & (times <= 3.5)] = 1.0

    islands = rms_islands(times, env, 0.9, min_sec=0.5, gap_sec=0.18)
    assert len(islands) == 1, islands
    start, _, end = islands[0]
    assert start <= 1.0 and end >= 3.5, islands


def test_rumble_filter_splits_ast_slab_to_loud_islands():
    """A 10 s AST vehicle span must not vibrate through a quiet gap."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 14.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.08 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    p1 = ((t >= 3.0) & (t <= 5.2)).astype(np.float32)
    p2 = ((t >= 9.0) & (t <= 11.0)).astype(np.float32)
    rumble = (0.55 * np.sin(2 * np.pi * 55 * t) * (p1 + p2)).astype(np.float32)
    audio = bed + rumble

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "slab.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [
            DetectedEvent("vehicle", "Vehicle", 1.0, 4.0, 12.0, 0.4),
        ]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert len(vehicles) >= 2
        assert all(e.end_sec - e.start_sec < 4.0 for e in vehicles)
        assert any(e.start_sec <= 3.3 and e.end_sec >= 4.8 for e in vehicles)
        assert any(e.start_sec <= 9.3 and e.end_sec >= 10.7 for e in vehicles)
        assert not any(e.start_sec <= 3.5 and e.end_sec >= 10.5 for e in vehicles)


def test_rumble_filter_keeps_steady_idle():
    """Unimodal engine bed is not carved into fake bursts."""
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 44100
    duration = 4.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = (0.3 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "idle.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        events = [DetectedEvent("vehicle", "Vehicle", 0.5, 2.0, 3.5, 0.4)]
        filtered = filter_sustained_rumble_bursts(events, wav, tax)
        vehicles = [e for e in filtered if e.category == "vehicle"]
        assert len(vehicles) == 1
        assert abs(vehicles[0].peak_sec - 2.0) < 0.05


def test_rumble_calibration_marks_fp_and_miss():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_calib import calibrate_rumble_thresholds

    sr = 44100
    duration = 10.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    bed = (0.1 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    burst = (
        0.5 * np.sin(2 * np.pi * 55 * t) * np.exp(-((t - 5.0) ** 2) / (2 * 0.08**2))
    ).astype(np.float32)
    audio = bed + burst
    events = {
        "events": [
            {
                "event_id": "event_001",
                "category": "vehicle",
                "peak_sec": 5.0,
                "confidence": 0.4,
                "audio_score": 0.38,
            },
            {
                "event_id": "event_002",
                "category": "vehicle",
                "peak_sec": 2.0,
                "confidence": 0.3,
                "audio_score": 0.28,
            },
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        report = calibrate_rumble_thresholds(
            wav, [5.0, 8.0], events, match_tolerance_sec=0.75
        )
    assert report["summary"]["hits"] == 1
    assert report["summary"]["misses"] == 1
    assert report["summary"]["false_positives"] == 1


def test_scan_rumble_timeline_100ms():
    import tempfile
    from pathlib import Path

    from haptic_gt.context.rumble_calib import scan_rumble_timeline

    sr = 44100
    duration = 6.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = (0.1 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        scan = scan_rumble_timeline(
            wav, start_sec=1.0, end_sec=2.0, hop_sec=0.1, manual_peaks_sec=[1.5]
        )
    assert scan["summary"]["n_samples"] >= 10
    ticks = [s["t_sec"] for s in scan["samples"]]
    assert any(abs(x - 1.5) < 1e-6 for x in ticks)


def test_intermittent_vehicle_masks_quiet_gaps():
    """Many vehicle spans → vibration on rumble windows, silence in gaps."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        _write_test_wav(source, duration=12.0)

        events = [
            DetectedEvent("vehicle", "Vehicle", 1.0, 1.2, 2.0, 0.4),
            DetectedEvent("vehicle", "Vehicle", 4.0, 4.2, 5.0, 0.4),
            DetectedEvent("vehicle", "Vehicle", 7.0, 7.2, 8.0, 0.4),
            DetectedEvent("vehicle", "Vehicle", 9.5, 9.7, 10.5, 0.4),
        ]

        def _passthrough(in_wav, out_wav):
            audio, sr = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, audio, sr, subtype="PCM_16")

        out = td_path / "haptic.wav"
        stitch_algorithm_output(
            source, events, out, _passthrough, continuous=ContinuousProfile()
        )
        haptic, sr = sf.read(out)
        gap = np.max(np.abs(haptic[int(2.5 * sr) : int(3.5 * sr)]))
        rumble = np.max(np.abs(haptic[int(1.2 * sr) : int(1.8 * sr)]))
        assert rumble > 0.05
        assert gap < rumble * 0.35


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
        assert tracks.algorithm_e is not None
        assert tracks.algorithm_e.exists()


def test_gate_report_names_the_scenes_and_the_spans_it_dropped():
    """A rumble that never reaches the haptic must say why, and how far off it was."""
    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 22050
    audio, _shots, drive, bursts = _combined_scenes_audio(sr)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "combined.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        proposals = [
            DetectedEvent("vehicle", "Vehicle", drive[0], 13.0, drive[1], 0.6),
            DetectedEvent("vehicle", "Vehicle", 18.4, 22.0, 29.0, 0.7),
            DetectedEvent("vehicle", "Vehicle", 8.6, 8.8, 9.4, 0.3),  # room tone
        ]
        report: dict = {}
        filter_sustained_rumble_bursts(proposals, wav, tax, report=report)

    assert report["proposals"] == 3
    assert report["kept"] >= 4, report
    assert len(report["scenes"]) >= 2, report["scenes"]
    quiet = [s for s in report["scenes"] if s["start_sec"] <= drive[0] < s["end_sec"]]
    loud = [s for s in report["scenes"] if s["start_sec"] <= bursts[0][0] < s["end_sec"]]
    assert quiet and loud
    assert quiet[0]["threshold"] < loud[0]["threshold"], report["scenes"]
    dropped = report["dropped"]
    assert any(d["local_rms"] < d["scene_threshold"] and d["reason"] for d in dropped), dropped


def test_a_close_car_rumble_does_not_gate_out_a_distant_drive():
    """One clip, three sources: the loudest must not set the floor for the rest."""
    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.taxonomy import load_taxonomy

    tax = load_taxonomy()
    sr = 22050
    audio, _shots, drive, bursts = _combined_scenes_audio(sr)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "combined.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        kept = filter_sustained_rumble_bursts(
            [
                DetectedEvent("vehicle", "Vehicle", drive[0], 13.0, drive[1], 0.6),
                DetectedEvent("vehicle", "Vehicle", 18.4, 22.0, 29.0, 0.7),
            ],
            wav,
            tax,
        )

    spans = [(e.start_sec, e.end_sec) for e in kept if e.category == "vehicle"]
    covered = sum(
        max(0.0, min(b, drive[1]) - max(a, drive[0])) for a, b in spans
    )
    assert covered / (drive[1] - drive[0]) > 0.7, spans
    car = [(a, b) for a, b in spans if a >= bursts[0][0] - 1.0]
    assert len(car) >= 3, spans
    assert all(b - a < 3.5 for a, b in car), car


def test_rumble_haptic_follows_the_bursts_inside_one_span():
    """A rumble's rhythm is all it has; a lull must come out clearly weaker."""
    sr = 22050
    n = int(10.0 * sr)
    t = np.arange(n) / sr
    quiet = 0.10 * np.sin(2 * np.pi * 45 * t)
    loud = 0.40 * np.sin(2 * np.pi * 45 * t) * (
        ((t >= 2.0) & (t <= 3.0)) | ((t >= 6.0) & (t <= 7.0))
    )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        sf.write(source, (quiet + loud).astype(np.float32), sr, subtype="PCM_16")

        def _passthrough(in_wav, out_wav):
            data, rate = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, data, rate, subtype="PCM_16")

        out = td_path / "haptic.wav"
        stitch_algorithm_output(
            source,
            [DetectedEvent("vehicle", "Vehicle", 0.5, 2.5, 9.5, 0.6)],
            out,
            _passthrough,
        )
        haptic, out_sr = sf.read(out)

    burst = _rms(haptic, out_sr, 2.2, 2.9)
    lull = _rms(haptic, out_sr, 4.2, 5.2)
    assert burst > 1.8 * lull, (burst, lull)
    assert lull > 0.005, lull


def test_a_distant_rumble_scene_hits_softer_than_a_close_one():
    """Spans are not all normalized to one peak, or the power is gone."""
    sr = 22050
    n = int(14.0 * sr)
    t = np.arange(n) / sr
    far = ((t >= 1.0) & (t <= 3.0)) | ((t >= 4.0) & (t <= 6.0))
    near = ((t >= 8.0) & (t <= 10.0)) | ((t >= 11.0) & (t <= 13.0))
    audio = 0.08 * np.sin(2 * np.pi * 45 * t) * far + 0.45 * np.sin(
        2 * np.pi * 45 * t
    ) * near
    spans = [(1.0, 3.0), (4.0, 6.0), (8.0, 10.0), (11.0, 13.0)]
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        sf.write(source, audio.astype(np.float32), sr, subtype="PCM_16")

        def _passthrough(in_wav, out_wav):
            data, rate = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, data, rate, subtype="PCM_16")

        out = td_path / "haptic.wav"
        stitch_algorithm_output(
            source,
            [DetectedEvent("vehicle", "Vehicle", a, (a + b) / 2, b, 0.6) for a, b in spans],
            out,
            _passthrough,
        )
        haptic, out_sr = sf.read(out)

    distant = _rms(haptic, out_sr, 1.3, 2.7)
    close = _rms(haptic, out_sr, 8.3, 9.7)
    assert distant < 0.75 * close, (distant, close)
    assert distant > 0.01, distant


def test_a_transient_in_one_span_does_not_make_that_span_the_quiet_one():
    """Rumble spans are levelled by energy; a spike must not crush the body."""
    sr = 22050
    n = int(14.0 * sr)
    t = np.arange(n) / sr
    spans = [(1.0, 3.0), (4.0, 6.0), (8.0, 10.0), (11.0, 13.0)]
    audio = np.zeros(n, dtype=np.float32)
    for a, b in spans:
        audio += (0.30 * np.sin(2 * np.pi * 45 * t) * ((t >= a) & (t <= b))).astype(
            np.float32
        )
    # One span opens on a clank, as a real rumble burst does
    i0 = int(spans[2][0] * sr)
    tt = np.arange(int(0.05 * sr)) / sr
    audio[i0 : i0 + tt.size] += (0.95 * np.exp(-tt * 120.0)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        sf.write(source, audio, sr, subtype="PCM_16")

        def _passthrough(in_wav, out_wav):
            data, rate = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, data, rate, subtype="PCM_16")

        out = td_path / "haptic.wav"
        stitch_algorithm_output(
            source,
            [DetectedEvent("vehicle", "Vehicle", a, (a + b) / 2, b, 0.6) for a, b in spans],
            out,
            _passthrough,
        )
        haptic, out_sr = sf.read(out)

    smooth = _rms(haptic, out_sr, 1.3, 2.7)
    with_clank = _rms(haptic, out_sr, 8.3, 9.7)
    assert 0.6 < with_clank / smooth < 1.7, (smooth, with_clank)


def test_accent_lands_slightly_ahead_of_the_attack():
    """A haptic aligned to the sample is felt late; it has to lead the bang."""
    sr = 22050
    n = int(4.0 * sr)
    audio = np.zeros(n, dtype=np.float32)
    peak_sec = 2.0
    i0 = int(peak_sec * sr)
    tt = np.arange(n - i0) / sr
    audio[i0:] = (0.9 * np.exp(-tt * 30.0) * np.sin(2 * np.pi * 80 * tt)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        sf.write(source, audio, sr, subtype="PCM_16")

        def _passthrough(in_wav, out_wav):
            data, rate = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, data, rate, subtype="PCM_16")

        out = td_path / "haptic.wav"
        profile = ContinuousProfile(enabled=False)
        stitch_algorithm_output(
            source,
            [DetectedEvent("explosion", "Explosion", 1.95, peak_sec, 2.6, 0.9)],
            out,
            _passthrough,
            continuous=profile,
        )
        haptic, out_sr = sf.read(out)

    strong = np.where(np.abs(haptic) >= 0.35 * np.max(np.abs(haptic)))[0]
    onset_sec = float(strong[0]) / out_sr
    lead = peak_sec - onset_sec
    assert 0.005 < lead < 0.060, (onset_sec, lead)


def test_volley_accents_do_not_ring_into_each_other():
    """Two bangs half a second apart must read as two hits, not one long buzz."""
    sr = 22050
    n = int(4.0 * sr)
    t = np.arange(n) / sr
    audio = (0.6 * np.sin(2 * np.pi * 70 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        source = td_path / "source.wav"
        sf.write(source, audio, sr, subtype="PCM_16")

        def _passthrough(in_wav, out_wav):
            data, rate = sf.read(in_wav, always_2d=False)
            sf.write(out_wav, data, rate, subtype="PCM_16")

        out = td_path / "haptic.wav"
        stitch_algorithm_output(
            source,
            [
                DetectedEvent("explosion", "Explosion", 1.0, 1.05, 1.4, 0.9),
                DetectedEvent("explosion", "Explosion", 1.5, 1.55, 1.9, 0.9),
            ],
            out,
            _passthrough,
            continuous=ContinuousProfile(enabled=False),
        )
        haptic, out_sr = sf.read(out)

    body = _rms(haptic, out_sr, 1.10, 1.30)
    # The first bang has to get out of the way before the second lands, or the
    # second attack has nothing to rise from.
    dip = min(
        _rms(haptic, out_sr, edge, edge + 0.01) for edge in np.arange(1.44, 1.53, 0.01)
    )
    assert dip < 0.4 * body, (body, dip)


def test_output_names_layout():
    assert OUTPUT_NAMES["algorithm_a_perception_mapping"].endswith(".wav")
    assert OUTPUT_NAMES["algorithm_e_rule_based"] == "algorithm_e_rule_based.wav"
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
