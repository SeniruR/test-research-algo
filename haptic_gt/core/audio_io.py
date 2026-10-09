"""Audio extraction, loading, and export (Sound2Hap-compatible rates)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

INPUT_SR = 44_100
VIB_SR = 8_000


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
    output_path: str | Path,
    sr: int = INPUT_SR,
) -> Path:
    """Extract mono 16-bit PCM WAV at 44.1 kHz from a video file."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return output_path


def prepare_source_wav(
    audio_path: str | Path,
    output_path: str | Path,
    sr: int = INPUT_SR,
) -> Path:
    """Convert/load audio to mono 16-bit PCM WAV at 44.1 kHz."""
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    return output_path


def save_haptic(path: str | Path, audio: np.ndarray, sr: int = VIB_SR) -> Path:
    """Write a mono haptic track as 16-bit PCM WAV (default 8 kHz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    return path
