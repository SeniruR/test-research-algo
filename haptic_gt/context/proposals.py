"""Cheap RMS onset proposals before frozen transformer classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.audio_io import INPUT_SR
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


@dataclass
class ProposalWindow:
    center_sec: float
    start_sec: float
    end_sec: float
    rms_score: float


def _envelope_rms(
    audio: np.ndarray,
    sr: int,
    *,
    hop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * hop_ms / 1000))
    n_frames = max(1, (len(audio) + hop - 1) // hop)
    env: list[float] = []
    times: list[float] = []
    for i in range(n_frames):
        s0 = i * hop
        s1 = min(len(audio), s0 + hop)
        chunk = audio[s0:s1]
        env.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))
        times.append((s0 + s1) / 2.0 / sr)
    return np.asarray(times, dtype=np.float64), np.asarray(env, dtype=np.float64)


def _local_peak_indices(
    values: np.ndarray,
    *,
    min_score: float,
    min_distance_frames: int,
) -> list[int]:
    candidates: list[int] = []
    n = len(values)
    for i in range(n):
        if values[i] < min_score:
            continue
        left = values[i - 1] if i > 0 else -1.0
        right = values[i + 1] if i + 1 < n else -1.0
        if values[i] >= left and values[i] >= right:
            candidates.append(i)

    candidates.sort(key=lambda i: values[i], reverse=True)
    kept: list[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_distance_frames for j in kept):
            kept.append(i)
    kept.sort()
    return kept


def merge_proposal_windows(
    windows: list[ProposalWindow],
    *,
    merge_gap_sec: float = 0.35,
) -> list[ProposalWindow]:
    """Merge overlapping or nearly-adjacent proposal windows."""
    if not windows:
        return []

    ordered = sorted(windows, key=lambda w: w.center_sec)
    merged: list[ProposalWindow] = [ordered[0]]
    for win in ordered[1:]:
        prev = merged[-1]
        if win.start_sec <= prev.end_sec + merge_gap_sec:
            merged[-1] = ProposalWindow(
                center_sec=win.center_sec
                if win.rms_score > prev.rms_score
                else prev.center_sec,
                start_sec=min(prev.start_sec, win.start_sec),
                end_sec=max(prev.end_sec, win.end_sec),
                rms_score=max(prev.rms_score, win.rms_score),
            )
        else:
            merged.append(win)
    return merged


def propose_onsets(
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    sample_rate: int = INPUT_SR,
) -> list[ProposalWindow]:
    """
    Find candidate transient times via short-hop RMS local maxima.

    Uses adaptive thresholding relative to the clip's RMS envelope.
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)

    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    if len(audio) == 0:
        return []

    hop_ms = taxonomy.proposal_rms_hop_ms
    pad_sec = taxonomy.proposal_search_pad_sec
    min_dist_sec = taxonomy.impulsive_min_peak_distance_sec

    times, env = _envelope_rms(audio, sr, hop_ms=hop_ms)
    if env.size == 0:
        return []

    peak_val = float(np.max(env))
    if peak_val < 1e-8:
        return []

    threshold = peak_val * taxonomy.proposal_threshold_ratio
    hop_sec = hop_ms / 1000.0
    min_dist_frames = max(1, int(round(min_dist_sec / hop_sec)))

    peak_idxs = _local_peak_indices(
        env,
        min_score=threshold,
        min_distance_frames=min_dist_frames,
    )

    duration_sec = len(audio) / sr
    windows: list[ProposalWindow] = []
    for idx in peak_idxs:
        center = float(times[idx])
        windows.append(
            ProposalWindow(
                center_sec=center,
                start_sec=max(0.0, center - pad_sec),
                end_sec=min(duration_sec, center + pad_sec),
                rms_score=float(env[idx]),
            )
        )

    return merge_proposal_windows(windows)


def propose_sustained_scan(
    duration_sec: float,
    taxonomy: Taxonomy | None = None,
    *,
    pad_sec: float | None = None,
) -> list[ProposalWindow]:
    """Uniform timeline windows for sustained categories (engine rumble, etc.)."""
    taxonomy = taxonomy or load_taxonomy()
    hop = taxonomy.proposal_sustained_hop_sec
    pad = pad_sec if pad_sec is not None else taxonomy.proposal_search_pad_sec
    if duration_sec <= 0 or hop <= 0:
        return []

    centers: list[float] = []
    t = hop / 2.0
    while t < duration_sec:
        centers.append(t)
        t += hop

    return [
        ProposalWindow(
            center_sec=c,
            start_sec=max(0.0, c - pad),
            end_sec=min(duration_sec, c + pad),
            rms_score=0.0,
        )
        for c in centers
    ]


def _dedupe_nearby_windows(
    windows: list[ProposalWindow],
    *,
    min_center_gap_sec: float,
) -> list[ProposalWindow]:
    """Keep highest-scoring window when centers are very close."""
    if not windows:
        return []

    ordered = sorted(windows, key=lambda w: w.center_sec)
    kept: list[ProposalWindow] = [ordered[0]]
    for win in ordered[1:]:
        if win.center_sec - kept[-1].center_sec < min_center_gap_sec:
            if win.rms_score > kept[-1].rms_score:
                kept[-1] = win
        else:
            kept.append(win)
    return kept


def propose_all_windows(
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    sample_rate: int = INPUT_SR,
    include_sustained_scan: bool = True,
) -> list[ProposalWindow]:
    """
    Onset transients plus optional sustained timeline scan for classifiers.

    Onset peaks are preserved; sustained windows fill gaps for vehicle / activity.
    """
    taxonomy = taxonomy or load_taxonomy()
    onset = propose_onsets(source_wav, taxonomy, sample_rate=sample_rate)
    if not include_sustained_scan:
        return onset

    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration_sec = len(audio) / (sr if sr else sample_rate)
    sustained = propose_sustained_scan(duration_sec, taxonomy)

    combined = _dedupe_nearby_windows(
        onset + sustained,
        min_center_gap_sec=0.35,
    )
    return combined
