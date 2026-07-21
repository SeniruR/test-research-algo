"""Peak/RMS/loudness normalization (from Sound2Hap)."""

from __future__ import annotations

import sys
import typing as tp

import torch
import torchaudio


def normalize_loudness(
    wav: torch.Tensor,
    sample_rate: int,
    loudness_headroom_db: float = 14,
    loudness_compressor: bool = False,
    energy_floor: float = 2e-3,
) -> torch.Tensor:
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    return output


def _clip_wav(
    wav: torch.Tensor,
    log_clipping: bool = False,
    stem_name: tp.Optional[str] = None,
) -> None:
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(
            f"CLIPPING {stem_name or ''} happening with proba:",
            clamp_prob,
            "maximum scale:",
            max_scale.item(),
            file=sys.stderr,
        )
    wav.clamp_(-1, 1)


def normalize_audio(
    wav: torch.Tensor,
    normalize: bool = True,
    strategy: str = "peak",
    peak_clip_headroom_db: float = 1,
    peak_normalize_db_clamp: float = 0,
    rms_headroom_db: float = 18,
    loudness_headroom_db: float = 14,
    loudness_compressor: bool = False,
    log_clipping: bool = False,
    sample_rate: tp.Optional[int] = None,
    stem_name: tp.Optional[str] = None,
) -> torch.Tensor:
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    normalize_peak = 10 ** (peak_normalize_db_clamp / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == "peak":
        wav_max = wav.abs().max()
        rescaling = (scale_peak / wav_max).clamp(max=(normalize_peak / wav_max).clamp(min=1))
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == "clip":
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == "rms":
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == "loudness":
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(
            wav, sample_rate, loudness_headroom_db, loudness_compressor
        )
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy in ("", "none"), f"Unexpected strategy: '{strategy}'"
    return wav
