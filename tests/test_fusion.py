"""Audio branch, video branch, and their fusion (no models loaded)."""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.components.detectors.fusion_detector.branches import (
    AudioBranchResult,
    VideoBranchResult,
    run_audio_branch,
    write_visual_context,
)
from haptic_gt.components.detectors.fusion_detector.encoders import EncoderScore
from haptic_gt.components.detectors.fusion_detector.frozen_fusion import DetectedEvent
from haptic_gt.components.detectors.fusion_detector.fusion import annotate_scene_labels, fuse_branches
from haptic_gt.components.detectors.fusion_detector.taxonomy import load_taxonomy
from haptic_gt.components.detectors.fusion_detector.visual_scenes import (
    VisualSpan,
    look_window,
    parse_categories,
    sample_fps,
)


def _shot_wav(td: str, shots=((1.0, 1.0), (2.5, 0.8)), sr: int = 22050) -> Path:
    t = np.arange(int(4.0 * sr), dtype=np.float32) / sr
    rng = np.random.default_rng(3)
    audio = (0.08 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    audio = audio + rng.standard_normal(t.size).astype(np.float32) * 0.05
    for peak_t, amp in shots:
        audio = audio + rng.standard_normal(t.size).astype(np.float32) * (
            amp * np.exp(-((t - peak_t) ** 2) / (2 * 0.010**2))
        )
    wav = Path(td) / "shots.wav"
    sf.write(wav, audio.astype(np.float32), sr, subtype="PCM_16")
    return wav


def _audio(events=(), candidates=()) -> AudioBranchResult:
    return AudioBranchResult(
        events=list(events),
        candidates=list(candidates),
        encoder_scores=[],
        gate_report={},
        detector_info={"mode": "frame_sed"},
    )


def _candidate(t: float) -> DetectedEvent:
    return DetectedEvent("explosion", "Explosion", t - 0.08, t, t + 0.45, 0.30, sources=["flux"])


def test_audio_branch_takes_no_video():
    params = inspect.signature(run_audio_branch).parameters
    assert list(params) == ["source_wav", "taxonomy"]


def test_scene_backs_an_unlabeled_sharp_attack():
    tax = load_taxonomy()
    video = VideoBranchResult(
        scenes=[VisualSpan("gunshot", 0.5, 2.0)], qwen_ran=True
    )
    events, report = fuse_branches(_audio(candidates=[_candidate(1.0)]), video, "unused.wav", tax)
    assert len(events) == 1
    assert events[0].category == "gunshot"
    assert abs(events[0].peak_sec - 1.0) < 1e-9
    assert "visual_scene" in events[0].sources
    assert report["scene_backed_added"] == 1


def test_long_scene_does_not_revive_the_war_tank_18s_phantom():
    """war_tank_001: the fireball stays on screen 14.2-18.7 s while the sound decays.

    Qwen labels that whole 4.4 s scene "explosion". The 18.09 s attack (sharpness
    5.8, under the 6.0 bar for an isolated end-of-clip hit) must stay out.
    """
    tax = load_taxonomy()
    scenes = [
        VisualSpan("explosion", 13.267, 14.233),
        VisualSpan("explosion", 14.233, 18.667),
        VisualSpan("explosion", 18.667, 18.7),
    ]
    video = VideoBranchResult(scenes=scenes, qwen_ran=True)
    events, report = fuse_branches(
        _audio(candidates=[_candidate(18.085)]), video, "unused.wav", tax
    )
    assert events == []
    assert report["scene_backed_added"] == 0


def test_car_crash_scene_backs_an_unlabeled_attack():
    tax = load_taxonomy()
    video = VideoBranchResult(
        scenes=[VisualSpan("car_crash", 0.5, 2.0), VisualSpan("smash", 0.5, 2.0)],
        qwen_ran=True,
    )
    events, report = fuse_branches(
        _audio(candidates=[_candidate(1.0)]), video, "unused.wav", tax
    )
    assert report["scene_backed_added"] == 1
    assert events[0].category == "car_crash"
    assert tax.categories["car_crash"].impulsive
    assert tax.categories["car_crash"].include_in_haptic_gate


def test_vehicle_or_engine_start_scenes_back_nothing():
    tax = load_taxonomy()
    for cat in ("vehicle", "engine_start"):
        video = VideoBranchResult(scenes=[VisualSpan(cat, 0.5, 2.0)], qwen_ran=True)
        events, report = fuse_branches(
            _audio(candidates=[_candidate(1.0)]), video, "unused.wav", tax
        )
        assert events == [], cat
        assert report["scene_backed_added"] == 0


def test_scene_alone_adds_nothing():
    tax = load_taxonomy()
    video = VideoBranchResult(scenes=[VisualSpan("explosion", 0.0, 4.0)], qwen_ran=True)
    events, report = fuse_branches(_audio(), video, "unused.wav", tax)
    assert events == []
    assert report["fused_events"] == 0


def test_scene_backing_needs_qwen_to_have_run():
    tax = load_taxonomy()
    video = VideoBranchResult(scenes=[VisualSpan("gunshot", 0.5, 2.0)], qwen_ran=False)
    events, report = fuse_branches(_audio(candidates=[_candidate(1.0)]), video, "unused.wav", tax)
    assert events == []
    assert report["scene_backing"] is False


def test_candidate_next_to_an_audio_bang_is_not_a_second_bang():
    tax = load_taxonomy()
    bang = DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.45, 0.4, sources=["audio"])
    video = VideoBranchResult(scenes=[VisualSpan("explosion", 0.5, 2.0)], qwen_ran=True)
    events, report = fuse_branches(
        _audio(events=[bang], candidates=[_candidate(1.2)]), video, "unused.wav", tax
    )
    assert len(events) == 1
    assert report["scene_backed_added"] == 0


def test_flash_snaps_adds_and_keeps_audio_bang_without_picture():
    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = _shot_wav(td, shots=((1.0, 1.0), (2.5, 0.8), (3.5, 0.8)))
        audio = _audio(events=[
            DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.45, 0.4, sources=["audio"]),
            # Off-screen: no flash, no scene. Must survive fusion.
            DetectedEvent("explosion", "Explosion", 3.42, 3.50, 3.95, 0.4, sources=["audio"]),
        ])
        video = VideoBranchResult(flashes=[1.02, 2.5])
        events, report = fuse_branches(audio, video, wav, tax)
    peaks = sorted(e.peak_sec for e in events if e.category == "explosion")
    assert any(abs(p - 3.5) < 0.1 for p in peaks), peaks
    assert any(abs(p - 2.5) < 0.1 for p in peaks), peaks
    assert any(abs(p - 1.0) < 0.1 for p in peaks), peaks
    assert report["flashes_added"] == 1
    assert report["flashes_snapped"] == 1


def test_scene_labels_are_recorded_without_moving_times():
    ev = DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.45, 0.4)
    out = annotate_scene_labels(
        [ev],
        [
            VisualSpan("smash", 0.0, 1.2),
            VisualSpan("gunshot", 0.0, 1.2),
            VisualSpan("vehicle", 5.0, 6.0),
        ],
    )
    assert out[0].visual_categories == ["smash", "gunshot"]
    assert (out[0].start_sec, out[0].peak_sec, out[0].end_sec) == (0.92, 1.00, 1.45)


def test_events_json_carries_fusion_and_scene_labels():
    from haptic_gt.components.detectors.fusion_detector.detector import EventResult

    ev = DetectedEvent("gunshot", "Gunshot, gunfire", 0.9, 1.0, 1.4, 0.4)
    ev.visual_categories = ["gunshot"]
    payload = EventResult(events=[ev], fusion={"fused_events": 1}).to_dict()
    assert payload["fusion"] == {"fused_events": 1}
    assert payload["events"][0]["visual_categories"] == ["gunshot"]
    assert "fusion" not in EventResult(events=[]).to_dict()


def test_visual_context_file_lists_flashes_and_spans():
    video = VideoBranchResult(
        flashes=[1.5],
        scenes=[VisualSpan("explosion", 1.0, 3.0)],
        qwen_ran=True,
    )
    with tempfile.TemporaryDirectory() as td:
        path = write_visual_context(video, td, video_name="clip.mp4")
        data = json.loads(path.read_text(encoding="utf-8"))
    assert data["video"] == "clip.mp4"
    assert data["flashes_sec"] == [1.5]
    assert data["spans"][0]["category"] == "explosion"


def test_promote_can_return_unbacked_candidates():
    from haptic_gt.components.detectors.fusion_detector.impulsive_promote import promote_impulsive_transients

    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = _shot_wav(td)
        events = [DetectedEvent("explosion", "Explosion", 0.92, 1.00, 1.45, 0.6)]
        scores = [EncoderScore(1.0, "Explosion", 0.6, "audio")]
        plain = promote_impulsive_transients(events, wav, scores, tax)
        both = promote_impulsive_transients(events, wav, scores, tax, return_candidates=True)
    assert isinstance(plain, list)
    kept, cands = both
    assert [e.peak_sec for e in kept] == [e.peak_sec for e in plain]
    assert all("flux" in c.sources for c in cands)


def test_parse_categories_keeps_only_taxonomy_names():
    assert parse_categories('{"categories": ["car_crash", "smash"]}') == [
        "car_crash",
        "smash",
    ]
    assert parse_categories('{"categories": ["engine start", "weather"]}') == ["engine_start"]
    assert parse_categories('{"categories": ["<name>"]}') == []
    assert parse_categories('{"category": "Explosion"}') == ["explosion"]
    assert parse_categories("there is no explosion here") == []
    assert parse_categories('{"categories": []}') == []


def test_scene_sampling_widens_tiny_cuts_and_caps_frames():
    assert look_window(10.0, 10.1, 60.0, min_look_sec=1.0) == (9.55, 10.55)
    assert look_window(59.9, 60.0, 60.0, min_look_sec=1.0) == (59.0, 60.0)
    assert look_window(10.0, 14.0, 60.0, min_look_sec=1.0) == (10.0, 14.0)
    assert sample_fps(0.0, 1.0, max_frames=16, min_look_sec=1.0) == 8.0
    assert sample_fps(0.0, 6.0, max_frames=16, min_look_sec=1.0) == 2.67
    assert sample_fps(0.0, 20.0, max_frames=16, min_look_sec=1.0) == 0.8
