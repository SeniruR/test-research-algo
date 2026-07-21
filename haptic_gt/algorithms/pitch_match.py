"""
Pitch Match audio-to-vibration (Sound2Hap / Kim et al., IEEE ToH 2023).

Python version used for Sound2Hap web tool. MATLAB version used in the study.

Adapted from: https://github.com/Iris1215/Sound2Hap
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample_poly

try:
    from mosqito.functions.loudness_zwtv._loudness_zwtv import loudness_zwtv

    MOSQITO_AVAILABLE = True
except Exception:
    MOSQITO_AVAILABLE = False

VIB_SR = 8000


@dataclass
class Config:
    regressionCoeffs: dict
    vibrationFreqRange: tuple
    binSizeMs: float
    overlapRatio: float
    smoothingWindow: int
    inputSampleRate: int
    outputSampleRate: int


def get_config() -> Config:
    return Config(
        regressionCoeffs={2: -0.005, 3: 0.003, 9: -0.015, 12: 0.008, 24: 0.008},
        vibrationFreqRange=(50.0, 398.0),
        binSizeMs=10.0,
        overlapRatio=0.5,
        smoothingWindow=3,
        inputSampleRate=44100,
        outputSampleRate=VIB_SR,
    )


def normalize_audio(audio: np.ndarray, do_normalize: bool) -> np.ndarray:
    scale_peak = 10 ** (-1 / 20)
    normalize_peak = 1.0
    wav_max = np.max(np.abs(audio)) + 1e-12
    rescaling = min(max(1.0, normalize_peak / wav_max), scale_peak / wav_max)
    if do_normalize or (rescaling < 1.0):
        audio = audio * rescaling
    return audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _specific_and_total_loudness_bark(audio_bin: np.ndarray, sr: int):
    if not MOSQITO_AVAILABLE:
        env = np.abs(audio_bin)
        total = float(np.mean(env))
        spec24 = np.zeros(24, dtype=np.float32)
        spec24[0] = total
        return spec24, total

    try:
        results = loudness_zwtv(audio_bin, sr, field_type="free")
        n_time = np.asarray(results["N"]).reshape(-1)
        n_spec = np.asarray(results["N_specific"])
        total_loudness = float(np.mean(n_time)) if n_time.size else 0.0

        if n_spec.ndim == 2 and n_spec.shape[1] >= 240:
            spec_time_mean = np.mean(n_spec, axis=0)
            spec24 = np.zeros(24, dtype=np.float32)
            for i in range(24):
                start = i * 10
                end = start + 10
                spec24[i] = float(np.sum(spec_time_mean[start:end]))
        else:
            if n_spec.ndim == 1:
                vec = n_spec
            else:
                vec = np.mean(n_spec, axis=0) if n_spec.size else np.zeros(240)
            idx = np.linspace(0, len(vec) - 1, 24)
            spec24 = np.interp(idx, np.arange(len(vec)), vec).astype(np.float32)

        spec24[~np.isfinite(spec24)] = 0.0
        return spec24, total_loudness
    except Exception:
        env = np.abs(audio_bin)
        total = float(np.mean(env))
        spec24 = np.zeros(24, dtype=np.float32)
        spec24[0] = total
        return spec24, total


def predict_vibration_frequency(specific_loudness_24: np.ndarray, cfg: Config) -> float:
    predicted = 0.0
    for bark_band, coeff in cfg.regressionCoeffs.items():
        idx = int(bark_band) - 1
        if 0 <= idx < len(specific_loudness_24):
            predicted += coeff * float(specific_loudness_24[idx]) * 1000.0
    predicted = abs(predicted)
    vmin, vmax = cfg.vibrationFreqRange
    return float(np.clip(predicted, vmin, vmax))


def analyze_audio_bins(audio: np.ndarray, sr: int, cfg: Config):
    bin_size = int(round(cfg.binSizeMs * sr / 1000.0))
    hop = max(1, int(round(bin_size * (1.0 - cfg.overlapRatio))))
    if bin_size < 2:
        bin_size = 2
    starts = np.arange(0, max(1, len(audio) - bin_size + 1), hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    times = (starts + bin_size / 2.0) / float(sr)
    freqs = np.zeros(starts.size, dtype=np.float32)
    amps = np.zeros(starts.size, dtype=np.float32)
    win = get_window("hann", bin_size, fftbins=False).astype(np.float32)

    for i, s in enumerate(starts):
        e = min(s + bin_size, len(audio))
        chunk = np.zeros(bin_size, dtype=np.float32)
        seg = audio[s:e]
        chunk[: len(seg)] = seg
        chunk *= win

        if rms(chunk) < 1e-3:
            freqs[i] = freqs[i - 1] if i > 0 else np.mean(cfg.vibrationFreqRange)
            amps[i] = 0.0
            continue

        spec24, loud = _specific_and_total_loudness_bark(chunk, sr)
        freqs[i] = predict_vibration_frequency(spec24, cfg)
        amps[i] = float(loud)

    if cfg.smoothingWindow > 1 and len(freqs) > cfg.smoothingWindow:
        k = cfg.smoothingWindow
        kernel = np.ones(k, dtype=np.float32) / k
        freqs = np.convolve(freqs, kernel, mode="same")

    return times.astype(np.float64), freqs.astype(np.float64), amps.astype(np.float64)


def generate_time_varying_vibration(audio: np.ndarray, sr: int, cfg: Config):
    bin_t, bin_f, bin_a = analyze_audio_bins(audio, sr, cfg)
    t = np.arange(len(audio), dtype=np.float64) / float(sr)

    if len(bin_t) == 1:
        f_inst = np.full_like(t, bin_f[0], dtype=np.float64)
        a_inst = np.full_like(t, bin_a[0], dtype=np.float64)
    else:
        try:
            from scipy.interpolate import PchipInterpolator

            f_inst = PchipInterpolator(bin_t, bin_f, extrapolate=True)(t)
        except Exception:
            f_inst = np.interp(t, bin_t, bin_f, left=bin_f[0], right=bin_f[-1])
        a_inst = np.interp(t, bin_t, bin_a, left=bin_a[0], right=bin_a[-1])

    rms_current = rms(a_inst)
    rms_norm = np.sqrt(2.0)
    rms_amplify = max(1.0, min(1.2, 1.0 / (rms_current * rms_norm))) if rms_current > 0 else 1.0
    a_inst = a_inst * (rms_norm * rms_amplify) if rms_current > 0 else np.full_like(t, 0.1)

    dt = 1.0 / float(sr)
    phi = np.empty_like(t)
    phi[0] = 0.0
    phi[1:] = 2.0 * np.pi * np.cumsum(f_inst[:-1]) * dt
    v = a_inst * np.sin(phi)

    fade_len = int(round(0.01 * sr))
    if len(v) > 2 * fade_len and fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        v[:fade_len] *= fade_in
        v[-fade_len:] *= fade_out

    return v.astype(np.float32), f_inst.astype(np.float32), a_inst.astype(np.float32)


def generate_vibration_signal(audio: np.ndarray, sr: int, cfg: Config):
    v, f_arr, a_arr = generate_time_varying_vibration(audio, sr, cfg)
    analysis_info = {
        "method": "time_varying",
        "duration": len(audio) / float(sr),
        "freqMean": float(np.mean(f_arr)),
        "freqRange": (float(np.min(f_arr)), float(np.max(f_arr))),
        "freqStd": float(np.std(f_arr)),
    }
    return v, f_arr, a_arr, analysis_info


def _read_mono(path: str | Path):
    x, sr = sf.read(path, always_2d=False)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr


def _write_int16_wav(path: str | Path, y: np.ndarray, sr: int):
    y = y / (np.max(np.abs(y)) + 1e-12)
    sf.write(path, y, sr, subtype="PCM_16")


def process_file(
    input_file: str | Path,
    output_file: str | Path,
    cfg: Config | None = None,
) -> dict:
    cfg = cfg or get_config()
    audio, sr = _read_mono(input_file)
    duration = len(audio) / float(sr)
    audio = normalize_audio(audio, True)

    v, f_arr, a_arr, info = generate_vibration_signal(audio, sr, cfg)
    fs_out = cfg.outputSampleRate
    if sr != fs_out:
        frac = Fraction(fs_out, sr).limit_denominator(1000)
        v = resample_poly(v, frac.numerator, frac.denominator)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_int16_wav(out_path, v, fs_out)

    return {
        "inputFile": str(input_file),
        "outputFile": str(output_file),
        "duration": duration,
        "originalSr": sr,
        "targetSr": fs_out,
        "analysisInfo": info,
        "mosqitoAvailable": MOSQITO_AVAILABLE,
    }
