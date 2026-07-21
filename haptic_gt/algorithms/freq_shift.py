"""
Frequency shifting audio-to-vibration (Sound2Hap / Okazaki et al., 2015).

Adapted from: https://github.com/Iris1215/Sound2Hap
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, lfilter

from haptic_gt.utils.normalization import normalize_audio

VIB_SR = 8000


def _butter_bandpass(sr: int, center_hz: float = 250.0, q: float = 1.0, order: int = 4):
    bw = center_hz / q
    low_hz = max(center_hz - bw / 2, 1.0)
    high_hz = min(center_hz + bw / 2, sr / 2 - 1)
    wn = [low_hz / (sr / 2), high_hz / (sr / 2)]
    return butter(order, wn, btype="band")


def _butter_highpass(sr: int, cutoff_hz: float = 10.0, order: int = 2):
    wn = cutoff_hz / (sr / 2)
    return butter(order, wn, btype="high")


def process_file(
    in_wav: Union[str, Path],
    out_wav: Union[str, Path],
    centre_hz: float = 250.0,
    q: float = 1.0,
) -> None:
    y, sr = librosa.load(in_wav, sr=None, mono=True)

    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
    y_norm_t = normalize_audio(wav_tensor, normalize=True, strategy="peak")
    y = y_norm_t.squeeze(0).numpy()

    y_1ot = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12, res_type="kaiser_best")
    y_2ot = librosa.effects.pitch_shift(y, sr=sr, n_steps=-24, res_type="kaiser_best")
    mix = y + y_1ot + y_2ot

    rms = np.sqrt(np.mean(mix**2) + 1e-12)
    mix /= rms * np.sqrt(2)

    b_hp, a_hp = _butter_highpass(sr, cutoff_hz=10.0)
    mix = lfilter(b_hp, a_hp, mix)

    b, a = _butter_bandpass(sr, centre_hz, q)
    mix_bp = lfilter(b, a, mix)
    mix_bp = librosa.resample(mix_bp, orig_sr=sr, target_sr=VIB_SR)
    mix_bp = np.clip(mix_bp, -1.0, 1.0)

    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix_bp.astype(np.float32), VIB_SR, subtype="PCM_16")
