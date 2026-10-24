"""Rapid fire: shots closer than the one-bang spacing (synthetic audio, no models)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.components.detectors.fusion_detector.frozen_fusion import DetectedEvent
from haptic_gt.components.detectors.fusion_detector.impulsive_nms import split_rapid_fire
from haptic_gt.components.detectors.fusion_detector.taxonomy import load_taxonomy

SR = 22050


def _wav(td: str, shots, duration: float = 7.0) -> Path:
    """Low noise bed plus a short noise burst per (time, amplitude)."""
    t = np.arange(int(duration * SR), dtype=np.float32) / SR
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(t.size).astype(np.float32) * 0.01
    for at, amp in shots:
        env = np.where(t >= at, np.exp(-(t - at) / 0.02), 0.0)
        audio = audio + rng.standard_normal(t.size).astype(np.float32) * (amp * env)
    wav = Path(td) / "burst.wav"
    sf.write(wav, audio.astype(np.float32), SR, subtype="PCM_16")
    return wav


def _shot(category, peak, end=None):
    return DetectedEvent(
        category=category,
        label=category,
        start_sec=peak - 0.08,
        peak_sec=peak,
        end_sec=peak + 0.45 if end is None else end,
        confidence=0.3,
        sources=["audio", "flux"],
    )


def _near(peaks, t, tol=0.03):
    return any(abs(p - t) <= tol for p in peaks)


def test_burst_every_third_of_a_second_becomes_separate_shots():
    burst = [1.0 + 0.33 * k for k in range(6)]  # 1.00 ... 2.65
    with tempfile.TemporaryDirectory() as td:
        wav = _wav(td, [(t, 0.8) for t in burst])
        out = split_rapid_fire([_shot("gunshot", 1.0)], wav, load_taxonomy())
    peaks = sorted(e.peak_sec for e in out)
    assert len(peaks) == len(burst)
    assert all(_near(peaks, t) for t in burst)
    assert sum("rapid_fire" in e.sources for e in out) == len(burst) - 1


def test_accents_keep_a_gap_so_the_burst_is_felt_as_pulses():
    tax = load_taxonomy()
    with tempfile.TemporaryDirectory() as td:
        wav = _wav(td, [(1.0, 0.8), (1.33, 0.8), (1.66, 0.8)])
        out = sorted(split_rapid_fire([_shot("gunshot", 1.0)], wav, tax), key=lambda e: e.peak_sec)
    for cur, nxt in zip(out, out[1:]):
        assert nxt.start_sec - cur.end_sec >= tax.impulsive_accent_gap_sec - 1e-6
        assert cur.end_sec > cur.peak_sec


def test_echoes_and_quiet_tails_are_not_shots():
    with tempfile.TemporaryDirectory() as td:
        # echo 0.16 s after each shot (inside the minimum gap), and a weak bump 0.5 s after
        wav = _wav(td, [(1.0, 0.8), (1.16, 0.5), (1.5, 0.15), (4.0, 0.8)])
        out = split_rapid_fire([_shot("gunshot", 1.0), _shot("gunshot", 4.0)], wav, load_taxonomy())
    assert sorted(round(e.peak_sec, 2) for e in out) == [1.0, 4.0]


def test_only_listed_categories_and_only_up_to_the_next_shot():
    with tempfile.TemporaryDirectory() as td:
        wav = _wav(td, [(1.0, 0.8), (1.33, 0.8), (4.0, 0.8), (4.33, 0.8), (6.0, 0.8)])
        events = [_shot("explosion", 1.0), _shot("gunshot", 4.0)]
        out = split_rapid_fire(events, wav, load_taxonomy())
    peaks = sorted(e.peak_sec for e in out)
    assert not _near(peaks, 1.33)  # explosion keeps the one-bang rule
    assert _near(peaks, 4.33)
    assert not _near(peaks, 6.0)  # beyond rapid_fire_window_sec of the last shot


def test_no_listed_category_leaves_events_untouched():
    with tempfile.TemporaryDirectory() as td:
        wav = _wav(td, [(1.0, 0.8), (1.33, 0.8)])
        events = [_shot("explosion", 1.0)]
        out = split_rapid_fire(events, wav, load_taxonomy())
    assert out == events and out[0].end_sec == 1.45
