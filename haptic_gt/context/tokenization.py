"""Phase 1 tokenization: 100Hz timeline and 16kHz audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MEL_SR = 16_000
TIMELINE_HZ = 100
FRAME_SIZE = 224


@dataclass
class TimelineTokens:
    """Aligned multimodal tokens on a fixed-rate timeline."""

    duration_sec: float
    timeline_hz: int
    video_frames: np.ndarray  # [T, 3, H, W] float32 in [0, 1]
    audio_16k: np.ndarray  # 1D float32 @ 16kHz
    source_fps: float


def _decode_video_frames(video_path: Path, n_bins: int, duration_sec: float) -> tuple[np.ndarray, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video tokenization. "
            "Install with: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        raw_frames.append(resized.astype(np.float32) / 255.0)

    cap.release()
    if not raw_frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    if duration_sec <= 0:
        duration_sec = len(raw_frames) / fps

    aligned = np.zeros((n_bins, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
    for i in range(n_bins):
        t = i / TIMELINE_HZ
        src_idx = int(round(t * fps))
        src_idx = min(max(src_idx, 0), len(raw_frames) - 1)
        aligned[i] = raw_frames[src_idx]

    video = np.transpose(aligned, (0, 3, 1, 2))
    return video, float(fps)


def _load_audio_16k(source_wav: Path) -> np.ndarray:
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != MEL_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=MEL_SR)
    return audio


def tokenize_video_audio(
    video_path: str | Path,
    source_wav: str | Path,
    *,
    timeline_hz: int = TIMELINE_HZ,
) -> TimelineTokens:
    """Build 100Hz-aligned video frames and 16kHz mono audio."""
    video_path = Path(video_path)
    source_wav = Path(source_wav)
    audio_16k = _load_audio_16k(source_wav)
    duration_sec = len(audio_16k) / MEL_SR
    n_bins = max(1, int(round(duration_sec * timeline_hz)))

    video_frames, source_fps = _decode_video_frames(video_path, n_bins, duration_sec)

    return TimelineTokens(
        duration_sec=duration_sec,
        timeline_hz=timeline_hz,
        video_frames=video_frames,
        audio_16k=audio_16k,
        source_fps=source_fps,
    )
