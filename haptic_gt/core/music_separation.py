"""Remove a music bed before detection and haptic generation.

The video the rater watches is unchanged. This only rewrites the WAV the
detector and the generators listen to. The original mix and the isolated
music are kept under debug/audio_preparation/.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from haptic_gt.core.audio_io import INPUT_SR

MIX_NAME = "original_mix.wav"
MUSIC_NAME = "music.wav"

SeparateFn = Callable[[Path, "MusicSeparationConfig"], tuple[np.ndarray, np.ndarray, int]]


@dataclass
class MusicSeparationConfig:
    enabled: bool = False
    #: Hugging Face id. Small fits a T4; large is better when memory allows.
    model: str = "facebook/sam-audio-small"
    #: What to pull out. The residual (everything else) is what the pipeline keeps.
    description: str = "music"
    #: Span prediction helps short events, not a music bed that runs under a scene.
    predict_spans: bool = False
    reranking_candidates: int = 1


def separate_music(
    source_wav: Path,
    debug_dir: Path,
    settings: MusicSeparationConfig,
    *,
    separate: SeparateFn | None = None,
) -> dict[str, Any]:
    """Replace ``source_wav`` with the non-music residual.

    ``debug_dir`` receives the untouched mix and the isolated music stem.
    ``separate`` is the model call; tests pass a stand-in.
    """
    source_wav = Path(source_wav)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    mix_path = debug_dir / MIX_NAME
    music_path = debug_dir / MUSIC_NAME
    shutil.copyfile(source_wav, mix_path)

    separate = separate or separate_with_sam_audio
    music, effects, sample_rate = separate(mix_path, settings)
    _write_wav(music_path, music, sample_rate)
    _write_wav(source_wav, effects, sample_rate)
    return {
        "model": settings.model,
        "description": settings.description,
        "predict_spans": settings.predict_spans,
        "reranking_candidates": settings.reranking_candidates,
        "sample_rate": int(sample_rate),
        "original_mix": mix_path.name,
        "music": music_path.name,
        "effects": source_wav.name,
    }


def separate_with_sam_audio(
    mix_path: Path,
    settings: MusicSeparationConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Isolate ``settings.description`` with SAM-Audio. Residual is the effects."""
    try:
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Music separation needs SAM-Audio. In Colab, after accepting the model "
            "terms on Hugging Face and running `hf auth login`:\n"
            "  pip install git+https://github.com/facebookresearch/sam-audio.git\n"
            "Then set music_separation.enabled to true."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMAudio.from_pretrained(settings.model).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(settings.model)
    batch = processor(audios=[str(mix_path)], descriptions=[settings.description]).to(device)
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=settings.predict_spans,
            reranking_candidates=settings.reranking_candidates,
        )
    music = _as_mono(result.target)
    effects = _as_mono(result.residual)
    return music, effects, int(processor.audio_sampling_rate)


def _as_mono(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import librosa

    if sample_rate != INPUT_SR:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=INPUT_SR)
    sf.write(path, np.clip(audio, -1.0, 1.0), INPUT_SR, subtype="PCM_16")
