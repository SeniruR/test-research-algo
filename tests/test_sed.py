"""Tests for frame-level SED decoding and DCASE-style metrics."""

from __future__ import annotations

import numpy as np

from haptic_gt.context.sed_events import (
    events_from_frame_posteriors,
    hysteresis_segments,
    median_filter,
    split_impulsive_by_posterior_peaks,
)
from haptic_gt.context.sed_frames import FramePosteriors, posteriors_to_encoder_scores
from haptic_gt.context.taxonomy import load_taxonomy
from haptic_gt.eval.sed_metrics import (
    RefEvent,
    evaluate_events,
    event_based_prf,
    format_evaluation,
    segment_based_prf,
)


def _frames(scores: dict[str, np.ndarray], hop: float = 0.1) -> FramePosteriors:
    n = len(next(iter(scores.values())))
    times = (np.arange(n) + 0.5) * hop
    return FramePosteriors(
        times=times,
        scores=scores,
        hop_sec=hop,
        backend="test",
        duration_sec=n * hop,
    )


def test_median_filter_removes_single_frame_spike():
    values = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
    out = median_filter(values, 3)
    assert out.max() < 0.2


def test_hysteresis_keeps_event_through_dip():
    values = np.array([0.0, 0.4, 0.2, 0.45, 0.0])
    segs = hysteresis_segments(values, high=0.35, low=0.15)
    assert len(segs) == 1
    assert segs[0] == (1, 3)


def test_hysteresis_ignores_below_high():
    values = np.array([0.0, 0.2, 0.25, 0.2, 0.0])
    assert hysteresis_segments(values, high=0.35, low=0.15) == []


def test_events_from_posteriors_timing_and_category():
    tax = load_taxonomy()
    n = 100
    veh = np.full(n, 0.05)
    veh[20:60] = 0.6  # 2.0 s -> 6.0 s
    exp = np.full(n, 0.02)
    exp[80:83] = 0.8  # ~8.0 s
    events = events_from_frame_posteriors(
        _frames({"vehicle": veh, "explosion": exp}), tax
    )
    cats = {e.category for e in events}
    assert cats == {"vehicle", "explosion"}
    vehicle = [e for e in events if e.category == "vehicle"][0]
    assert abs(vehicle.start_sec - 2.0) < 0.15, vehicle.start_sec
    assert abs(vehicle.end_sec - 6.0) < 0.15, vehicle.end_sec
    blast = [e for e in events if e.category == "explosion"][0]
    assert abs(blast.peak_sec - 8.05) < 0.2, blast.peak_sec
    assert "sed" in blast.sources
    assert blast.video_score is None


def test_events_from_posteriors_min_duration():
    tax = load_taxonomy()
    n = 60
    exp = np.full(n, 0.02)
    exp[30] = 0.9
    events = events_from_frame_posteriors(_frames({"explosion": exp}), tax)
    assert len(events) == 1
    assert events[0].end_sec - events[0].start_sec >= tax.sed_min_event_sec_impulsive


def test_split_impulsive_volley_into_separate_shots():
    tax = load_taxonomy()
    n = 160
    exp = np.full(n, 0.05)
    # Continuous above-threshold volley with three posterior peaks
    exp[95:150] = 0.4
    for center in (100, 115, 140):
        exp[center] = 0.85
    frames = _frames({"explosion": exp})
    events = events_from_frame_posteriors(frames, tax)
    assert len(events) == 1
    split = split_impulsive_by_posterior_peaks(events, frames, tax)
    peaks = sorted(e.peak_sec for e in split)
    assert len(peaks) == 3, peaks
    for want in (10.05, 11.55, 14.05):
        assert any(abs(p - want) < 0.2 for p in peaks), (want, peaks)


def test_posteriors_to_encoder_scores_uses_mappable_labels():
    from haptic_gt.context.taxonomy import match_label_to_category

    tax = load_taxonomy()
    frames = _frames({"vehicle": np.array([0.4, 0.5]), "explosion": np.array([0.0, 0.7])})
    scores = posteriors_to_encoder_scores(frames, tax)
    assert scores
    for s in scores:
        assert match_label_to_category(tax, s.label, s.source) in ("vehicle", "explosion")


def test_event_based_prf_collar_and_one_to_one():
    ref = {"explosion": [10.0, 11.5]}
    dets = [
        {"category": "explosion", "peak_sec": 10.05, "end_sec": 10.5},
        {"category": "explosion", "peak_sec": 10.12, "end_sec": 10.6},  # duplicate
    ]
    report = event_based_prf(ref, dets, onset_collar_sec=0.2)
    assert report["micro"]["tp"] == 1
    assert report["micro"]["fp"] == 1
    assert report["micro"]["fn"] == 1
    assert abs(report["mean_abs_onset_error_sec"] - 0.05) < 1e-6


def test_event_based_prf_wide_collar_recovers_hand_marks():
    """Hand marks carry ~0.5 s of timing error; a wider collar should show it."""
    ref = {"explosion": [9.64, 11.16]}
    dets = [
        {"category": "explosion", "peak_sec": 10.12},
        {"category": "explosion", "peak_sec": 11.00},
    ]
    tight = event_based_prf(ref, dets, onset_collar_sec=0.2)
    wide = event_based_prf(ref, dets, onset_collar_sec=0.5)
    assert tight["micro"]["tp"] == 1
    assert wide["micro"]["tp"] == 2
    assert wide["micro"]["f1"] == 1.0


def test_segment_based_prf_ignores_onset_jitter():
    ref = [RefEvent("vehicle", 1.0, 4.0)]
    dets = [{"category": "vehicle", "start_sec": 1.2, "end_sec": 3.9}]
    report = segment_based_prf(ref, dets, duration_sec=6.0, segment_sec=1.0)
    assert report["micro"]["f1"] > 0.8


def test_sed_chain_on_mixed_tank_and_cannon_clip():
    """Decode -> split -> flux refine -> promote -> rumble filter, end to end.

    Posteriors are given 0.15 s late on purpose: frame-level SED localizes to its
    frame grid, and spectral-flux refinement must pull onsets back to the attack.
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from haptic_gt.context.frozen_fusion import dedupe_events_by_peak
    from haptic_gt.context.impulsive_promote import promote_impulsive_transients
    from haptic_gt.context.onset_refine import refine_event_timing
    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts
    from haptic_gt.context.sed_frames import posteriors_to_encoder_scores
    from haptic_gt.context.sustained_merge import merge_sustained_events

    tax = load_taxonomy()
    sr = 44100
    duration = 16.0
    shots = [9.64, 11.16, 11.66, 12.45, 13.50, 14.55]
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(21)

    drive_gate = ((t >= 0.3) & (t <= 8.8)).astype(np.float32)
    audio = 0.22 * np.sin(2 * np.pi * 42 * t) * drive_gate
    for clank_t in np.arange(0.6, 8.8, 0.7):
        clank = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + clank * (0.06 * np.exp(-((t - clank_t) ** 2) / (2 * 0.006**2)))
    for shot_t in shots:
        blast = rng.standard_normal(t.shape[0]).astype(np.float32)
        audio = audio + blast * (0.95 * np.exp(-((t - shot_t) ** 2) / (2 * 0.012**2)))

    hop = 0.1
    n = int(duration / hop)
    times = (np.arange(n) + 0.5) * hop
    veh = np.where((times >= 0.3) & (times <= 8.8), 0.55, 0.05)
    exp = np.full(n, 0.03)
    for shot_t in shots:
        idx = int(round((shot_t + 0.15) / hop))
        exp[min(idx, n - 1)] = 0.85
    frames = FramePosteriors(
        times=times,
        scores={"vehicle": veh, "explosion": exp},
        hop_sec=hop,
        backend="test",
        duration_sec=duration,
    )

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mixed.wav"
        sf.write(wav, audio.astype(np.float32), sr, subtype="PCM_16")

        events = events_from_frame_posteriors(frames, tax)
        events = split_impulsive_by_posterior_peaks(events, frames, tax)
        events = refine_event_timing(events, wav, tax)
        events = dedupe_events_by_peak(events)
        events = promote_impulsive_transients(
            events, wav, posteriors_to_encoder_scores(frames, tax), tax
        )
        events = refine_event_timing(events, wav, tax)
        events = dedupe_events_by_peak(events)
        events = merge_sustained_events(events, tax)
        events = filter_sustained_rumble_bursts(events, wav, tax)
        events = dedupe_events_by_peak(events)

    rows = [
        {
            "category": e.category,
            "peak_sec": e.peak_sec,
            "start_sec": e.start_sec,
            "end_sec": e.end_sec,
        }
        for e in events
    ]
    scored = event_based_prf(
        {"explosion": shots}, rows, onset_collar_sec=0.2, categories=["explosion"]
    )
    assert scored["micro"]["recall"] == 1.0, scored["matches"]
    assert scored["micro"]["fp"] <= 1, scored["matches"]
    # Flux refinement, not the 100 ms posterior grid, sets the onset
    assert scored["mean_abs_onset_error_sec"] < 0.05, scored["mean_abs_onset_error_sec"]

    vehicles = [e for e in events if e.category == "vehicle"]
    assert vehicles, "tank drive must survive as rumble"
    assert any(e.start_sec <= 2.0 and e.end_sec >= 4.0 for e in vehicles), [
        (round(e.start_sec, 2), round(e.end_sec, 2)) for e in vehicles
    ]
    assert all(e.start_sec < 9.2 for e in vehicles), [
        (round(e.start_sec, 2), round(e.end_sec, 2)) for e in vehicles
    ]

    # A continuous drive has to feel continuous: a handful of spans covering most
    # of it, not a stutter of short buzzes.
    spans = sorted(((e.start_sec, e.end_sec) for e in vehicles))
    assert len(spans) <= 3, spans
    covered = sum(min(e, 8.8) - max(s, 0.3) for s, e in spans if e > 0.3 and s < 8.8)
    assert covered >= 0.70 * (8.8 - 0.3), (covered, spans)
    # Overlapping spans would buzz twice over the same audio
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans


def test_evaluate_events_report_formats():
    ref = {"explosion": [10.0], "vehicle": [2.0]}
    events = {
        "events": [
            {"category": "explosion", "peak_sec": 10.1, "start_sec": 10.0, "end_sec": 10.5},
            {"category": "vehicle", "peak_sec": 2.4, "start_sec": 2.0, "end_sec": 4.0},
        ]
    }
    report = evaluate_events(ref, events, duration_sec=12.0, collars_sec=(0.2, 0.5))
    assert len(report["event_based"]) == 2
    text = format_evaluation(report)
    assert "Event-based F1" in text
    assert "Segment-based F1" in text
    assert "explosion" in text


def _war_clip_audio(sr: int = 22050, quiet_middle: bool = True):
    """Tank drive with a quieter camera angle, then a cannon volley.

    Mirrors the measured profile of the real war clip: drive attacks peak near
    0.34 of clip max, blasts run 0.42-1.00, and one volley shot lands in the
    decay of the loudest blast.
    """
    duration = 19.0
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(7)

    level = np.where((t >= 4.2) & (t <= 7.5), 0.09, 0.20).astype(np.float32)
    if not quiet_middle:
        level = np.full_like(t, 0.20)
    audio = level * np.sin(2 * np.pi * 45 * t) * ((t >= 0.3) & (t <= 9.0))
    for clank_t in np.arange(0.6, 9.0, 0.5):
        amp = 0.05 if not (4.2 <= clank_t <= 7.5) else 0.02
        audio = audio + rng.standard_normal(t.size).astype(np.float32) * (
            amp * np.exp(-((t - clank_t) ** 2) / (2 * 0.006**2))
        )

    blasts = {9.6: 0.25, 10.12: 0.55, 11.0: 0.70, 11.52: 1.0, 12.24: 0.55, 13.4: 0.9, 15.0: 0.5}
    for shot_t, amp in blasts.items():
        audio = audio + rng.standard_normal(t.size).astype(np.float32) * (
            amp * np.exp(-((t - shot_t) ** 2) / (2 * 0.013**2))
        )
        # Boom decay: what crushes prominence for the next shot in the volley
        tail = (t >= shot_t) & (t <= shot_t + 1.2)
        audio = audio + (
            0.45 * amp * np.sin(2 * np.pi * 70 * t) * np.exp(-(t - shot_t) / 0.35) * tail
        )
    return audio.astype(np.float32), duration, sorted(blasts)


def test_volley_shot_in_a_decay_tail_is_still_found():
    """A blast 0.7 s after a louder one has no prominence left; level must carry it.

    Prominence compares a peak to the level just before it, and the previous
    boom is still ringing there. Without the volley rule this shot is dropped.
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from haptic_gt.context.encoders import EncoderScore
    from haptic_gt.context.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    sr = 22050
    audio, _duration, blasts = _war_clip_audio(sr)

    scores = [
        EncoderScore(time_sec=t, label="Explosion", score=0.42, source="audio")
        for t in np.arange(9.4, 16.0, 0.2)
    ]

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "war.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        out = promote_impulsive_transients([], wav, scores, tax)

    peaks = sorted(e.peak_sec for e in out if e.category in ("explosion", "gunshot"))
    assert any(abs(p - 12.24) <= 0.25 for p in peaks), peaks
    # The drive must stay free of bangs even though clanks are loud
    assert not [p for p in peaks if p < 9.2], peaks


def test_quieter_camera_angle_stays_part_of_the_same_rumble():
    """A scene cut to a wider shot halves the level while the tank keeps rolling.

    The clip has one loudness mode, so the gate must sit under its quietest
    sustained part; sitting at half the median silences the wide-angle stretch.
    """
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from haptic_gt.context.frozen_fusion import DetectedEvent
    from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts

    tax = load_taxonomy()
    sr = 22050
    audio, _duration, blasts = _war_clip_audio(sr)

    events = [DetectedEvent("vehicle", "Vehicle", 1.4, 2.3, 9.0, 0.6)]
    events += [
        DetectedEvent("explosion", "Explosion", t - 0.08, t, t + 0.85, 0.42)
        for t in blasts
    ]

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "war.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")
        out = filter_sustained_rumble_bursts(events, wav, tax)

    spans = sorted((e.start_sec, e.end_sec) for e in out if e.category == "vehicle")
    assert spans, "drive must survive"
    covered = sum(min(e, 9.0) - max(s, 1.4) for s, e in spans if e > 1.4 and s < 9.0)
    assert covered >= 0.70 * (9.0 - 1.4), (covered, spans)


def test_sustained_marks_are_scored_by_coverage_not_onset_f1():
    """A long rumble span covering every mark must not score 0 on onset collar.

    The marks sit inside the burst, so onset matching and segment presence both
    punish a correct detection. Coverage is the honest read.
    """
    ref = {"explosion": [10.0], "vehicle": [29.0, 30.0, 31.0]}
    events = {
        "events": [
            {"category": "explosion", "peak_sec": 10.05, "start_sec": 9.9, "end_sec": 10.4},
            {"category": "vehicle", "peak_sec": 30.2, "start_sec": 28.5, "end_sec": 32.0},
        ]
    }
    report = evaluate_events(
        ref,
        events,
        duration_sec=60.0,
        collars_sec=(0.2,),
        sustained_categories=("vehicle",),
    )
    block = report["event_based"][0]
    assert "vehicle" not in block["per_category"]
    assert block["per_category"]["explosion"]["f1"] == 1.0

    cov = report["coverage"]["per_category"]["vehicle"]
    assert cov["covered"] == 3
    assert cov["recall"] == 1.0
    assert cov["uncovered_marks"] == []
    assert cov["active_fraction"] < 0.10

    text = format_evaluation(report)
    assert "Sustained coverage" in text


def test_coverage_flags_marks_outside_detected_spans():
    ref = {"vehicle": [6.0, 24.0, 48.0]}
    events = {"events": [{"category": "vehicle", "start_sec": 5.0, "end_sec": 7.0, "peak_sec": 6.1}]}
    report = evaluate_events(
        ref, events, duration_sec=60.0, sustained_categories=("vehicle",)
    )
    cov = report["coverage"]["per_category"]["vehicle"]
    assert cov["covered"] == 1
    assert [m["mark_sec"] for m in cov["uncovered_marks"]] == [24.0, 48.0]
    assert cov["uncovered_marks"][0]["distance_to_span_sec"] == 17.0


def test_coverage_tolerates_marks_rounded_just_before_a_burst():
    """A mark typed as "24" for a burst starting at 24.06 is not a miss."""
    ref = {"vehicle": [6.0, 24.0, 51.0]}
    events = {
        "events": [
            {"category": "vehicle", "start_sec": 6.338, "end_sec": 7.416, "peak_sec": 6.73},
            {"category": "vehicle", "start_sec": 24.063, "end_sec": 25.669, "peak_sec": 24.23},
            {"category": "vehicle", "start_sec": 51.191, "end_sec": 55.631, "peak_sec": 51.25},
        ]
    }
    report = evaluate_events(
        ref, events, duration_sec=72.0, sustained_categories=("vehicle",)
    )
    cov = report["coverage"]["per_category"]["vehicle"]
    assert cov["covered"] == 3, cov
    assert cov["recall"] == 1.0


def test_report_says_nothing_to_score_instead_of_printing_zeros():
    """A clip with no impulsive marks must not read as an F1 of 0."""
    ref = {"vehicle": [10.0]}
    events = {"events": [{"category": "vehicle", "start_sec": 9.0, "end_sec": 12.0, "peak_sec": 10.2}]}
    report = evaluate_events(
        ref, events, duration_sec=30.0, sustained_categories=("vehicle",)
    )
    text = format_evaluation(report)
    assert "nothing to score" in text
    assert "Event-based F1 (onset collar" not in text
    assert "Sustained coverage" in text


if __name__ == "__main__":
    test_median_filter_removes_single_frame_spike()
    test_hysteresis_keeps_event_through_dip()
    test_events_from_posteriors_timing_and_category()
    test_split_impulsive_volley_into_separate_shots()
    test_event_based_prf_collar_and_one_to_one()
    print("All SED tests passed.")
