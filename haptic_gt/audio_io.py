"""Audio extraction, loading, and export."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 44_100


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. Install it first "
            "(Colab: !apt-get -qq install ffmpeg)."
        )
    return ffmpeg


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sr: int = TARGET_SR,
) -> tuple[np.ndarray, int]:
    """
    Extract mono audio from a video file at the given sample rate.

    Returns (audio, sample_rate). If output_path is set, also writes a WAV file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = _require_ffmpeg()
    wav_path = Path(output_path) if output_path else video_path.with_suffix(".wav")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    audio, loaded_sr = librosa.load(wav_path, sr=sr, mono=True)
    return audio.astype(np.float32), loaded_sr


def load_audio(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load mono audio from a WAV/MP3/etc. file."""
    audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32), loaded_sr


def save_haptic(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> Path:
    """Write a mono haptic track as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(path, clipped, sr, subtype="PCM_16")
    return path
