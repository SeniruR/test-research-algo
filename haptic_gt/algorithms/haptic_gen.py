"""
HapticGen-style RMS-driven NCO synthesis (Sound2Hap / Sung et al., CHI 2025).

Adapted from: https://github.com/Iris1215/Sound2Hap
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from haptic_gt.utils.normalization import normalize_audio

WANTED_BIN_SIZE_SEC = 0.010
BASE_FREQ = 200.0
VIB_SR = 8000


def amp_env_on_wav_norm(
    wav_norm: np.ndarray,
    input_sample_rate: int,
    output_sample_rate: int,
) -> np.ndarray:
    wav_norm = wav_norm.squeeze()
    num_samples = len(wav_norm)
    duration_sec = num_samples / input_sample_rate
    samples_per_bin = int(WANTED_BIN_SIZE_SEC * input_sample_rate)
    num_bins = num_samples // samples_per_bin

    wav_chunks = np.array_split(wav_norm, num_bins)
    rms_bins = np.array([np.sqrt(np.mean(chunk**2)) for chunk in wav_chunks])
    rms_max = np.max(rms_bins)
    rms_norm = np.sqrt(2)
    rms_amplify = max(1.0, min(1.2, 1.0 / (rms_max * rms_norm)))
    rms_norm_amp = rms_norm * rms_amplify
    out_samples = int(duration_sec * output_sample_rate)

    phase_acc = 0.0
    output = np.zeros(out_samples)
    for i in range(out_samples):
        t = i / output_sample_rate
        t_prog = t / duration_sec
        bin_fi = t_prog * num_bins
        bin_lo = int(bin_fi)
        bin_hi = min(num_bins - 1, int(math.ceil(bin_fi)))
        bin_fr = bin_fi - bin_lo
        rms_val = (
            rms_bins[bin_lo] * (1.0 - bin_fr) + rms_bins[bin_hi] * bin_fr
        ) * rms_norm_amp
        freq_offset = (rms_val - 0.3) * 100.0
        phase_delta = 2.0 * math.pi * (BASE_FREQ + freq_offset) / output_sample_rate
        phase_acc = (phase_acc + phase_delta) % (2.0 * math.pi)
        output[i] = rms_val * math.sin(phase_acc)

    return output


def process_file(input_path: str | Path, output_path: str | Path) -> None:
    wav_data, sr = sf.read(input_path)
    wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
    wav_norm_tensor = normalize_audio(
        wav_tensor,
        normalize=True,
        strategy="peak",
        peak_clip_headroom_db=0,
        peak_normalize_db_clamp=0,
    )
    wav = wav_norm_tensor.squeeze(0).numpy()
    env_signal = amp_env_on_wav_norm(wav, sr, VIB_SR)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, env_signal, VIB_SR, subtype="PCM_16")
