"""Tests for the original rule-based haptic mapper (algorithm E)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.algorithms.rule_based import (
    MIN_INTENSITY,
    mix_window,
    process_file,
    windows_to_wav,
)


def test_mix_window_thunder_both_is_full_scale():
    kind, intensity = mix_window(max_lightning=150.0, max_rain=0.0, avg_audio=120.0)
    assert kind == "thunder_both"
    assert intensity == 255


def test_mix_window_quiet_audio_is_silent():
    kind, intensity = mix_window(max_lightning=0.0, max_rain=0.0, avg_audio=float(MIN_INTENSITY - 1))
    assert kind == "none"
    assert intensity == 0


def test_mix_window_rain_beats_soft_audio():
    kind, intensity = mix_window(max_lightning=0.0, max_rain=80.0, avg_audio=40.0)
    assert kind == "rain"
    assert intensity == 80


def test_process_file_follows_loud_burst():
    sr = 44100
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    audio[(t >= 0.8) & (t < 1.0)] = 0.9
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.wav"
        out = Path(td) / "e.wav"
        js = Path(td) / "e.json"
        sf.write(src, audio, sr, subtype="PCM_16")
        payload = process_file(src, out, json_path=js)
        assert out.exists()
        haptic, out_sr = sf.read(out)
        assert out_sr == 8000
        assert abs(len(haptic) / out_sr - 2.0) < 0.05

        track = json.loads(js.read_text())["track"]
        quiet = [int(v) for k, v in track.items() if int(k) < 700]
        loud = [int(v) for k, v in track.items() if 800 <= int(k) < 1000]
        assert max(quiet) == 0
        assert max(loud) >= 200

        env = windows_to_wav(payload["events"], 2.0)
        assert env.max() > 0.5


def test_wav_is_silent_when_map_is_zero():
    events = [
        {"start_ms": 0, "type": "none", "intensity": 0},
        {"start_ms": 40, "type": "none", "intensity": 0},
    ]
    wav = windows_to_wav(events, 0.08)
    assert float(np.max(np.abs(wav))) < 1e-6
