"""scripts/trim_outputs.py: find a trimmed clip's cuts and carry the outputs over."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "trim_outputs", Path(__file__).resolve().parents[1] / "scripts" / "trim_outputs.py"
)
trim_outputs = importlib.util.module_from_spec(_spec)
sys.modules["trim_outputs"] = trim_outputs
_spec.loader.exec_module(trim_outputs)

SR = trim_outputs.ALIGN_SR


def _scene(seconds: float, seed: int = 4) -> np.ndarray:
    """Noise whose loudness changes every 0.2 s, like a soundtrack's envelope."""
    rng = np.random.default_rng(seed)
    steps = rng.uniform(0.02, 1.0, int(seconds / 0.2) + 1)
    gain = np.repeat(steps, int(0.2 * SR))[: int(seconds * SR)]
    return (rng.standard_normal(gain.size) * gain * 0.3).astype(np.float32)


def _trim(full: np.ndarray, kept: list[tuple[float, float]]) -> np.ndarray:
    return np.concatenate([full[int(a * SR) : int(b * SR)] for a, b in kept])


def test_finds_cuts_and_offsets_of_a_trimmed_clip():
    full = _scene(120.0)
    trim = _trim(full, [(0.0, 30.0), (45.0, 70.0), (100.0, 115.0)])
    segments, coverage = trim_outputs.find_segments(trim, full)
    assert coverage > 0.9
    assert [round(s.offset, 3) for s in segments] == [0.0, 15.0, 45.0]
    assert segments[1].start == pytest.approx(30.0, abs=0.05)
    assert segments[2].start == pytest.approx(55.0, abs=0.05)
    assert segments[-1].end == pytest.approx(70.0, abs=1e-3)


def test_unrelated_audio_is_refused():
    full = _scene(120.0, seed=4)
    other = _scene(60.0, seed=9)
    _, coverage = trim_outputs.find_segments(other, full)
    assert coverage < trim_outputs.MIN_COVERAGE


def test_tracks_and_events_follow_the_cuts():
    segments = [
        trim_outputs.Segment(0.0, 30.0, 0.0),
        trim_outputs.Segment(30.0, 55.0, 15.0),
    ]
    track = np.arange(80 * 10, dtype=np.float32)  # 10 Hz "WAV", value = sample index
    cut = trim_outputs.cut_track(track, 10, segments, 55.0)
    assert cut.size == 550
    assert cut[299] == 299 and cut[300] == 450  # 30 s in the trim plays 45 s of the full clip

    events = [
        {"event_id": "event_001", "start_sec": 10.0, "peak_sec": 10.1, "end_sec": 10.5},
        {"event_id": "event_002", "start_sec": 35.0, "peak_sec": 35.1, "end_sec": 35.5},  # cut out
        {"event_id": "event_003", "start_sec": 49.9, "peak_sec": 50.0, "end_sec": 50.4},
    ]
    out = trim_outputs.cut_events(events, segments)
    assert [(e["event_id"], e["peak_sec"], e["source_peak_sec"]) for e in out] == [
        ("event_001", 10.1, 10.1),
        ("event_002", 35.0, 50.0),
    ]
    assert out[1]["start_sec"] == 34.9
