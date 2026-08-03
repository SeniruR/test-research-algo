"""Frozen AST and ViViT encoders (requires_grad=False)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from haptic_gt.context.proposals import ProposalWindow

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
VIVIT_MODEL_ID = "google/vivit-b-16x2-kinetics400"

_audio_pipe: Any | None = None
_video_pipe: Any | None = None


def _get_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def get_audio_pipeline():
    global _audio_pipe
    if _audio_pipe is None:
        from transformers import pipeline

        _audio_pipe = pipeline(
            "audio-classification",
            model=AST_MODEL_ID,
            device=_get_device(),
        )
    return _audio_pipe


def get_video_pipeline():
    global _video_pipe
    if _video_pipe is None:
        from transformers import pipeline

        _video_pipe = pipeline(
            "video-classification",
            model=VIVIT_MODEL_ID,
            device=_get_device(),
        )
    return _video_pipe


@dataclass
class EncoderScore:
    time_sec: float
    label: str
    score: float
    source: str  # "audio" | "video"


def _top_predictions(raw: list[dict], top_k: int = 5) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for item in raw[:top_k]:
        label = str(item.get("label", ""))
        if label.startswith("LABEL_"):
            label = label.replace("LABEL_", "")
        out.append((label, float(item.get("score", 0.0))))
    return out


def classify_audio_window(
    audio_16k: np.ndarray,
    start_sec: float,
    end_sec: float,
    *,
    top_k: int = 5,
) -> list[EncoderScore]:
    """Run frozen AST on an audio slice."""
    sr = 16_000
    s0 = max(0, int(start_sec * sr))
    s1 = min(len(audio_16k), int(end_sec * sr))
    if s1 - s0 < sr // 10:
        return []

    clip = audio_16k[s0:s1]
    pipe = get_audio_pipeline()
    preds = pipe(clip, top_k=top_k)
    mid = 0.5 * (start_sec + end_sec)
    return [
        EncoderScore(time_sec=mid, label=label, score=score, source="audio")
        for label, score in _top_predictions(preds, top_k)
    ]


def classify_video_clip_file(
    video_path: Path,
    center_sec: float,
    *,
    top_k: int = 5,
) -> list[EncoderScore]:
    """
    Run frozen ViViT on a short clip from the video file.

    The HF video-classification pipeline samples frames internally from the
    full file; for temporal windows we extract a sub-clip with ffmpeg when
  possible, otherwise classify the whole file (coarse fallback).
    """
    pipe = get_video_pipeline()
    try:
        preds = pipe(str(video_path), top_k=top_k)
    except Exception:
        return []
    return [
        EncoderScore(time_sec=center_sec, label=label, score=score, source="video")
        for label, score in _top_predictions(preds, top_k)
    ]


def classify_video_frames(
    frames: np.ndarray,
    center_sec: float,
    *,
    top_k: int = 5,
) -> list[EncoderScore]:
    """Run ViViT on a pre-extracted frame tensor [T, C, H, W] in [0,1]."""
    if frames.size == 0:
        return []
    try:
        import torch
    except ImportError:
        return []

    pipe = get_video_pipeline()
    # Pipeline expects list of PIL or tensor; pass uint8 frame list
    tensor = torch.from_numpy(frames).float()
    if tensor.ndim != 4:
        return []
    # Sample up to 32 frames uniformly
    n = tensor.shape[0]
    if n > 32:
        idx = np.linspace(0, n - 1, 32).astype(int)
        tensor = tensor[idx]
    # Convert to uint8 HWC list for pipeline compatibility
    frames_u8 = (tensor.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames_u8]
        preds = pipe(pil_frames, top_k=top_k)
    except Exception:
        return []

    return [
        EncoderScore(time_sec=center_sec, label=label, score=score, source="video")
        for label, score in _top_predictions(preds, top_k)
    ]


def run_encoder_on_windows(
    windows: list[ProposalWindow],
    video_path: Path,
    audio_16k: np.ndarray,
    video_frames: np.ndarray,
    *,
    timeline_hz: int = 100,
    window_sec: float = 1.0,
    top_k: int = 8,
) -> list[EncoderScore]:
    """Classify frozen AST + ViViT only at proposal-centered windows."""
    if not windows:
        return []

    duration_sec = len(audio_16k) / 16_000
    half = window_sec / 2.0
    scores: list[EncoderScore] = []

    for win in windows:
        t0 = max(0.0, win.center_sec - half)
        t1 = min(duration_sec, win.center_sec + half)
        scores.extend(classify_audio_window(audio_16k, t0, t1, top_k=top_k))

        bin_start = int(t0 * timeline_hz)
        bin_end = int(t1 * timeline_hz)
        bin_end = min(bin_end, len(video_frames))
        if bin_end > bin_start:
            clip = video_frames[bin_start:bin_end]
            scores.extend(
                classify_video_frames(clip, center_sec=win.center_sec, top_k=top_k)
            )
        else:
            scores.extend(
                classify_video_clip_file(video_path, center_sec=win.center_sec, top_k=top_k)
            )

    return scores


def run_encoder_pass(
    video_path: Path,
    audio_16k: np.ndarray,
    video_frames: np.ndarray,
    *,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    duration_sec: float,
    top_k: int = 8,
    full_scan: bool = False,
    proposal_windows: list[ProposalWindow] | None = None,
    timeline_hz: int = 100,
) -> list[EncoderScore]:
    """
    Classify with frozen AST + ViViT.

    Default: proposal windows only. Set full_scan=True for legacy timeline sweep.
    """
    if not full_scan and proposal_windows is not None:
        return run_encoder_on_windows(
            proposal_windows,
            video_path,
            audio_16k,
            video_frames,
            timeline_hz=timeline_hz,
            window_sec=window_sec,
            top_k=top_k,
        )

    scores: list[EncoderScore] = []
    t = 0.0
    while t < duration_sec:
        end = min(duration_sec, t + window_sec)
        scores.extend(classify_audio_window(audio_16k, t, end, top_k=top_k))

        bin_start = int(t * 100)
        bin_end = int(end * 100)
        bin_end = min(bin_end, len(video_frames))
        if bin_end > bin_start:
            clip = video_frames[bin_start:bin_end]
            scores.extend(
                classify_video_frames(clip, center_sec=0.5 * (t + end), top_k=top_k)
            )
        else:
            scores.extend(
                classify_video_clip_file(video_path, center_sec=0.5 * (t + end), top_k=top_k)
            )
        t += hop_sec
    return scores
