"""
Perception-level audio-to-vibration translator (Sound2Hap / Lee & Choi, CHI 2013).

Adapted from: https://github.com/Iris1215/Sound2Hap
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import find_peaks

from haptic_gt.utils.normalization import normalize_audio

AUDIO_SR = 44100
VIB_SR = 8000
FRAME_S = 4096
F1 = 175.0
F2 = 210.0
CR = 0.035
OR = 0.40
CV = 1
CL = 0.1
OL = 3.8
C_FULLBAND = 0.065
F_FULLBAND = 6400
C_BASS = 1.91
F_BASS = 200
C = 1.37

_ISO_FREQ = np.array(
    [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    ]
)
_ISO_SPL60 = np.array(
    [
        104.23, 99.08, 94.18, 89.96, 85.94, 82.05, 78.65, 75.56, 72.47, 69.86,
        67.53, 65.39, 63.45, 62.05, 60.81, 59.89, 60.01, 62.15, 63.19, 59.96,
        57.26, 56.42, 57.57, 60.89, 66.36,
    ]
)


def iso60phon(f: np.ndarray) -> np.ndarray:
    return np.interp(f, _ISO_FREQ, _ISO_SPL60, left=_ISO_SPL60[0], right=_ISO_SPL60[-1])


def auditory_loudness(frame: np.ndarray, content: str) -> float:
    if content == "music":
        c_use, f_max = C_BASS, F_BASS
    else:
        c_use, f_max = C_FULLBAND, F_FULLBAND

    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame.size, 1 / AUDIO_SR)
    mask = (freqs >= 25) & (freqs <= f_max)
    mag = mag[mask]
    freqs = freqs[mask]

    db = 20 * np.log10(C * mag + 1e-12)
    af = iso60phon(freqs)
    loudness = c_use * np.sum(db / af)
    return max(0.0, loudness)


def auditory_roughness(frame: np.ndarray, peak_db: float = -40.0) -> float:
    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame.size, 1 / AUDIO_SR)
    mask = (freqs >= 25) & (freqs <= 6400)
    mag = mag[mask]
    freqs = freqs[mask]

    db = 20 * np.log10(mag + 1e-12)
    thresh = db.max() + peak_db
    peaks, _ = find_peaks(db, height=thresh)

    f = freqs[peaks]
    x = mag[peaks]
    roughness = 0.0
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            f1, f2 = f[i], f[j]
            x1, x2 = x[i], x[j]
            xm, xM = min(x1, x2), max(x1, x2)
            fd = abs(f2 - f1)
            s = 0.24 / (0.0207 * min(f1, f2) + 18.96)
            term = ((xm * xM) ** 0.1 / 2.0) * (2 * xm / (xm + xM)) ** 3.11
            roughness += term * (math.exp(-3.5 * s * fd) - math.exp(-5.75 * s * fd))
    return roughness


def perceptual_targets(la: float, ra: float, content: str) -> tuple[float, float]:
    if content == "music":
        iv = CL * la - OL
    else:
        iv = CR * math.sqrt(la) * (ra**2) - OR
    rv = CV * ra
    return max(0, iv), rv


def amplitudes_from_percepts(iv: float, rv: float) -> tuple[float, float]:
    if iv <= 0.0:
        return 0.0, 0.0

    rv_max = (801.0 / 113.0) + 0.529 * iv + 0.479
    rv_adj = min(rv, rv_max)
    disc = max(0.0, 801.0 - 113.0 * (rv_adj - 0.529 * iv - 0.479))
    r1 = (28.3 + math.sqrt(disc)) / 56.3
    r2 = (28.3 - math.sqrt(disc)) / 56.3
    valid = [s for s in (r1, r2) if 0.0 <= s <= 1.0]
    s = min(valid) if valid else (28.3 / 56.3)

    a = ((25.8 * s**2 - 25.5 * s + rv_adj - 0.203) / 3.98) ** 2
    a2 = a * s
    a1 = a - a2
    return a1, a2


def synth_vibration(a1: float, a2: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / VIB_SR
    return a1 * np.sin(2 * math.pi * F1 * t) + a2 * np.sin(2 * math.pi * F2 * t)


def read_wav_mono_44k(fname: str | Path) -> np.ndarray:
    wav_data, _ = sf.read(fname)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)
    wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
    wav_norm_tensor = normalize_audio(
        wav_tensor,
        normalize=True,
        strategy="peak",
        peak_clip_headroom_db=0,
        peak_normalize_db_clamp=0,
    )
    return wav_norm_tensor.squeeze(0).numpy().astype("float32")


def process_file(
    in_wav: str | Path,
    out_wav: str | Path,
    content: str = "game",
) -> None:
    audio = read_wav_mono_44k(in_wav)
    hop_s = FRAME_S
    n_out_total = int(np.ceil(len(audio) * VIB_SR / AUDIO_SR))
    vib_full = np.zeros(n_out_total, dtype=np.float32)

    for start in range(0, len(audio), hop_s):
        block = audio[start : start + FRAME_S]
        if block.size == 0:
            break
        if block.size < FRAME_S:
            block = np.pad(block, (0, FRAME_S - block.size), "constant")

        la = auditory_loudness(block, content)
        ra = auditory_roughness(block)
        iv, rv = perceptual_targets(la, ra, content)
        a1, a2 = amplitudes_from_percepts(iv, rv)

        n_out = int(round(FRAME_S * VIB_SR / AUDIO_SR))
        vib_seg = synth_vibration(a1, a2, n_out)
        rms_seg = np.sqrt(np.mean(vib_seg**2) + 1e-12)
        vib_seg /= rms_seg * np.sqrt(2)

        out_start = int(round(start * VIB_SR / AUDIO_SR))
        out_end = out_start + n_out
        if out_end > n_out_total:
            vib_full[out_start:] += vib_seg[: n_out_total - out_start]
        else:
            vib_full[out_start:out_end] += vib_seg

    vib_full = np.clip(vib_full, -1.0, 1.0)
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, vib_full, VIB_SR, subtype="PCM_16")
