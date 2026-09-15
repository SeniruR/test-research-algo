"""Original rule-based haptic map (thunder / rain / audio RMS).

Port of ``vibrator-android/model_json/process_video.py``. Unlike Sound2Hap A–D
this runs on the ungated mix (and video frames when present), then writes a
40 ms intensity map plus an 8 kHz carrier WAV the Android A–E switcher can play.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from haptic_gt.audio_io import VIB_SR

WINDOW_SIZE_MS = 40
CARRIER_HZ = 175.0

LIGHTNING_BRIGHTNESS_SCALE_MAX = 50.0
LIGHTNING_THRESHOLD_VISUAL_LOW = 140.0
LIGHTNING_THRESHOLD_VISUAL_HIGH = 160.0
RAIN_DIFF_THRESHOLD = 20
RAIN_RATIO_MAX = 0.12
RAIN_INTENSITY_MAX = 200.0
RAIN_INTENSITY_THRESHOLD = 30.0
AUDIO_RMS_MAX = 0.5
AUDIO_THUNDER_INTENSITY_THRESHOLD = 100.0
MIN_INTENSITY = 16


def mix_window(
    *,
    max_lightning: float,
    max_rain: float,
    avg_audio: float,
) -> tuple[str, int]:
    """Choose event type and 0–255 intensity for one 40 ms window."""
    event_type = "none"
    intensity = 0

    if max_lightning > LIGHTNING_THRESHOLD_VISUAL_LOW and avg_audio > AUDIO_THUNDER_INTENSITY_THRESHOLD:
        event_type = "thunder_both"
        intensity = 255
    elif max_lightning > LIGHTNING_THRESHOLD_VISUAL_HIGH:
        event_type = "thunder_visual"
        intensity = 255
    elif avg_audio > AUDIO_THUNDER_INTENSITY_THRESHOLD and max_lightning > LIGHTNING_THRESHOLD_VISUAL_LOW:
        event_type = "thunder_audio"
        intensity = int(min(255, avg_audio * 1.5))
    elif max_rain > RAIN_INTENSITY_THRESHOLD:
        event_type = "rain"
        intensity = int(max_rain)
    elif avg_audio > MIN_INTENSITY:
        event_type = "audio"
        intensity = int(avg_audio)
    else:
        event_type = "none"
        intensity = 0

    if intensity < MIN_INTENSITY:
        intensity = 0
        if event_type != "none":
            event_type = "none"
    return event_type, int(intensity)


def _audio_intensity(rms: float) -> float:
    return float(np.interp(rms, [0.0, AUDIO_RMS_MAX], [0.0, 255.0]))


def _rms_windows(audio: np.ndarray, sr: int, window_ms: int = WINDOW_SIZE_MS) -> np.ndarray:
    hop = max(1, int(round(sr * window_ms / 1000.0)))
    n = int(np.ceil(len(audio) / hop))
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        chunk = audio[i * hop : min(len(audio), (i + 1) * hop)]
        if chunk.size:
            out[i] = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
    return out


def _analyze_audio_only(audio: np.ndarray, sr: int) -> list[dict[str, Any]]:
    rms = _rms_windows(audio, sr)
    events: list[dict[str, Any]] = []
    for i, val in enumerate(rms):
        avg_audio = _audio_intensity(float(val))
        event_type, intensity = mix_window(
            max_lightning=0.0, max_rain=0.0, avg_audio=avg_audio
        )
        events.append(
            {
                "start_ms": int(i * WINDOW_SIZE_MS),
                "type": event_type,
                "intensity": intensity,
                "max_lightning": 0.0,
                "max_rain": 0.0,
                "avg_audio": float(avg_audio),
            }
        )
    return events


def _analyze_video(
    video_path: Path,
    audio: np.ndarray,
    sr: int,
) -> list[dict[str, Any]] | None:
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-6 or total_frames <= 0 or frame_width <= 0:
        cap.release()
        return None

    frame_length = max(1, int(round(sr / fps)))
    audio_rms = librosa.feature.rms(
        y=audio.astype(np.float32),
        frame_length=frame_length,
        hop_length=frame_length,
    )[0]

    frames_per_window = max(1, int(fps * (WINDOW_SIZE_MS / 1000.0)))
    last_brightness = 0.0
    prev_down = None
    frame_metrics: list[dict[str, float]] = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        down = cv2.resize(gray, (max(1, frame_width // 4), max(1, frame_height // 4)))
        if prev_down is None:
            prev_down = down
        diff = cv2.absdiff(down, prev_down)
        prev_down = down
        _, diff_th = cv2.threshold(diff, RAIN_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        rain_motion = int(np.count_nonzero(diff_th))

        avg_brightness = float(np.mean(gray))
        brightness_spike = max(0.0, avg_brightness - last_brightness)
        last_brightness = avg_brightness
        lightning_intensity = float(
            np.interp(
                brightness_spike,
                [0.0, LIGHTNING_BRIGHTNESS_SCALE_MAX],
                [0.0, 255.0],
            )
        )
        rain_ratio = rain_motion / float(down.shape[0] * down.shape[1])
        rain_intensity = float(
            np.interp(rain_ratio, [0.0, RAIN_RATIO_MAX], [0.0, RAIN_INTENSITY_MAX])
        )
        audio_val = float(audio_rms[frame_idx]) if frame_idx < len(audio_rms) else 0.0
        frame_metrics.append(
            {
                "lightning_intensity": lightning_intensity,
                "rain_intensity": rain_intensity,
                "audio_intensity": _audio_intensity(audio_val),
                "brightness": avg_brightness,
            }
        )

    cap.release()
    if not frame_metrics:
        return None

    num_windows = int(np.ceil(len(frame_metrics) / frames_per_window))
    events: list[dict[str, Any]] = []
    for w in range(num_windows):
        window_frames = frame_metrics[w * frames_per_window : (w + 1) * frames_per_window]
        if not window_frames:
            events.append(
                {
                    "start_ms": int(w * WINDOW_SIZE_MS),
                    "type": "none",
                    "intensity": 0,
                    "max_lightning": 0.0,
                    "max_rain": 0.0,
                    "avg_audio": 0.0,
                }
            )
            continue
        max_lightning = max(f["lightning_intensity"] for f in window_frames)
        max_rain = max(f["rain_intensity"] for f in window_frames)
        avg_audio = float(np.mean([f["audio_intensity"] for f in window_frames]))
        event_type, intensity = mix_window(
            max_lightning=max_lightning, max_rain=max_rain, avg_audio=avg_audio
        )
        events.append(
            {
                "start_ms": int(w * WINDOW_SIZE_MS),
                "type": event_type,
                "intensity": intensity,
                "max_lightning": float(max_lightning),
                "max_rain": float(max_rain),
                "avg_audio": avg_audio,
            }
        )
    return events


def windows_to_track(events: list[dict[str, Any]]) -> dict[str, int]:
    return {str(int(e["start_ms"])): int(e["intensity"]) for e in events}


def windows_to_wav(
    events: list[dict[str, Any]],
    duration_sec: float,
    *,
    sr: int = VIB_SR,
    carrier_hz: float = CARRIER_HZ,
) -> np.ndarray:
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    carrier = np.sin(2.0 * np.pi * carrier_hz * t)
    amp = np.zeros(n, dtype=np.float64)
    hop = max(1, int(round(sr * WINDOW_SIZE_MS / 1000.0)))
    for i, ev in enumerate(events):
        s0 = i * hop
        s1 = min(n, (i + 1) * hop)
        if s1 <= s0:
            break
        amp[s0:s1] = float(ev["intensity"]) / 255.0
    if events:
        last_end = min(n, len(events) * hop)
        if last_end < n:
            amp[last_end:] = float(events[-1]["intensity"]) / 255.0
    return (carrier * amp).astype(np.float32)


def analyze(
    source_wav: str | Path,
    *,
    video_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    events = None
    if video_path is not None:
        events = _analyze_video(Path(video_path), audio, int(sr))
    if events is None:
        events = _analyze_audio_only(audio, int(sr))
    return events


def process_file(
    source_wav: str | Path,
    output_wav: str | Path,
    *,
    video_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the rule-based haptic WAV (and optional JSON map)."""
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration_sec = len(audio) / float(sr)

    events = analyze(source_wav, video_path=video_path)
    haptic = windows_to_wav(events, duration_sec)
    out_wav = Path(output_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, haptic, VIB_SR, subtype="PCM_16")

    payload = {
        "window_size_ms": WINDOW_SIZE_MS,
        "track": windows_to_track(events),
        "events": events,
        "source": "rule_based",
    }
    if json_path is not None:
        out_json = Path(json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["json_path"] = str(out_json)
    payload["wav_path"] = str(out_wav)
    return payload
