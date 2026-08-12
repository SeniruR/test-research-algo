"""Refine event peak/start/end using spectral-flux onset alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def _envelope_rms(
    audio: np.ndarray,
    sr: int,
    *,
    hop_ms: float = 5.0,
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


def _spectral_flux(
    audio: np.ndarray,
    sr: int,
    *,
    hop_ms: float = 5.0,
    n_fft: int = 512,
    hf_weight: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Positive spectral flux with high-frequency emphasis.

    Better than RMS for impulsive onsets (gunshot / explosion attacks).
    """
    hop = max(1, int(sr * hop_ms / 1000))
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(audio) - n_fft) // hop)
    if n_frames < 2:
        return np.array([0.0]), np.array([0.0])

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # Emphasize mid/high bands typical of blast attacks
    band_w = np.ones_like(freqs, dtype=np.float32)
    band_w[freqs >= 500.0] = hf_weight
    band_w[freqs >= 2000.0] = hf_weight * 1.25
    band_w[freqs < 80.0] = 0.35

    mags: list[np.ndarray] = []
    times: list[float] = []
    for i in range(n_frames):
        s0 = i * hop
        s1 = s0 + n_fft
        if s1 > len(audio):
            break
        frame = audio[s0:s1] * window
        mag = np.abs(np.fft.rfft(frame)).astype(np.float32) * band_w
        mags.append(mag)
        times.append((s0 + n_fft / 2.0) / sr)

    if len(mags) < 2:
        return np.asarray(times, dtype=np.float64), np.zeros(len(times), dtype=np.float64)

    flux = np.zeros(len(mags), dtype=np.float64)
    for i in range(1, len(mags)):
        diff = mags[i] - mags[i - 1]
        flux[i] = float(np.sum(np.maximum(diff, 0.0)))

    return np.asarray(times, dtype=np.float64), flux


def local_flux_ratio(
    times: np.ndarray,
    flux: np.ndarray,
    peak_t: float,
    *,
    pre_sec: float = 0.65,
    gap_sec: float = 0.04,
    min_pre_sec: float = 0.20,
    max_ratio: float = 999.0,
) -> float:
    """Peak flux vs median flux just before it (sharp attack vs rumble).

    Returns 0 without enough pre-context: the clip's first frames always look
    like a huge attack (silence to signal) and must not count as a transient.
    After digital silence the baseline is exactly 0, so the ratio is capped
    rather than allowed to blow up to 1e14.
    """
    if flux.size == 0 or times.size == 0:
        return 0.0
    idx = int(np.argmin(np.abs(times - peak_t)))
    peak_v = float(flux[idx])
    pre_mask = (times >= peak_t - pre_sec) & (times < peak_t - gap_sec)
    pre = flux[pre_mask]
    pre_times = times[pre_mask]
    if pre.size == 0:
        return 0.0
    if float(pre_times[-1] - pre_times[0]) < min_pre_sec:
        return 0.0
    baseline = float(np.median(pre))
    if baseline <= 0.0:
        baseline = float(np.mean(pre))
    if baseline <= 0.0:
        return max_ratio if peak_v > 0.0 else 0.0
    return min(peak_v / baseline, max_ratio)


def _pick_onset_from_flux(
    times: np.ndarray,
    flux: np.ndarray,
    *,
    min_ratio: float = 0.45,
    prefer_earliest: bool = False,
    early_rel: float = 0.55,
) -> float | None:
    """Pick onset from flux peaks.

    Default: strongest peak, earliest among near-ties.
    ``prefer_earliest``: earliest peak that is still comparable to the strongest
    one in the window (``early_rel``). A weak earlier bump — a track clank
    before the shot — must not steal the onset.
    """
    if flux.size == 0:
        return None
    peak = float(np.max(flux))
    if peak < 1e-12:
        return None

    floor = peak * min_ratio
    candidates: list[int] = []
    for i in range(1, len(flux) - 1):
        if flux[i] < floor:
            continue
        if flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            candidates.append(i)
    if not candidates:
        return float(times[int(np.argmax(flux))])

    if prefer_earliest:
        best_val = max(float(flux[i]) for i in candidates)
        pool = [i for i in candidates if float(flux[i]) >= best_val * early_rel]
        earliest = min(pool or candidates, key=lambda i: float(times[i]))
        return float(times[earliest])

    # Prefer the highest flux; among near-ties, earliest attack (muzzle), not later rumble
    best_val = max(float(flux[i]) for i in candidates)
    strong = [i for i in candidates if float(flux[i]) >= best_val * 0.92]
    earliest = min(strong, key=lambda i: float(times[i]))
    return float(times[earliest])


def _find_acoustic_peak(
    audio: np.ndarray,
    sr: int,
    center_sec: float,
    radius_sec: float,
) -> float:
    """Peak RMS sample time near the classifier's rough center."""
    s0 = max(0, int((center_sec - radius_sec) * sr))
    s1 = min(len(audio), int((center_sec + radius_sec) * sr))
    if s1 <= s0:
        return center_sec

    seg = audio[s0:s1]
    times, env = _envelope_rms(seg, sr)
    if env.size == 0:
        return center_sec
    peak_idx = int(np.argmax(env))
    return float(s0 / sr + times[peak_idx])


def _find_impulsive_peak(
    audio: np.ndarray,
    sr: int,
    center_sec: float,
    taxonomy: Taxonomy,
    *,
    duration_sec: float,
) -> float:
    """
    Snap impulsive events to spectral-flux onset near the classifier hint.

    Uses high-frequency-weighted spectral flux (not RMS max), which better
    matches gunshot / explosion attacks. Falls back to RMS if flux is weak.

    On short clips only, look back to t=0 so a late AST/ViViT hit (e.g. rumble)
    can still snap to an earlier muzzle blast. Mixed clips keep a local window
    so tank-drive clanks are not stolen as the cannon onset.
    """
    back = taxonomy.impulsive_onset_back_sec
    forward = taxonomy.impulsive_onset_forward_sec
    if duration_sec <= taxonomy.impulsive_short_clip_sec:
        back = max(back, center_sec)
    hop_ms = taxonomy.onset_flux_hop_ms
    s0 = max(0, int((center_sec - back) * sr))
    s1 = min(len(audio), int((center_sec + forward) * sr))
    if s1 <= s0:
        return center_sec

    seg = audio[s0:s1]
    times, flux = _spectral_flux(seg, sr, hop_ms=hop_ms)
    onset_rel = _pick_onset_from_flux(
        times,
        flux,
        min_ratio=taxonomy.onset_flux_min_ratio,
        prefer_earliest=True,
        early_rel=taxonomy.onset_flux_early_rel,
    )
    if onset_rel is not None:
        return float(s0 / sr + onset_rel)

    # Fallback: RMS peak in the same window
    times_rms, env = _envelope_rms(seg, sr, hop_ms=max(2.5, hop_ms))
    if env.size == 0:
        return center_sec
    return float(s0 / sr + times_rms[int(np.argmax(env))])


def _extend_decay_tail(
    audio: np.ndarray,
    sr: int,
    peak_sec: float,
    *,
    threshold_ratio: float,
    max_tail_sec: float,
    duration_sec: float,
) -> float:
    """Extend gate end while RMS stays above a fraction of the peak."""
    peak_idx = int(peak_sec * sr)
    hop = max(1, int(sr * 0.005))
    lo = max(0, peak_idx - hop)
    hi = min(len(audio), peak_idx + hop)
    peak_val = float(np.sqrt(np.mean(audio[lo:hi] ** 2) + 1e-12))
    if peak_val < 1e-8:
        return peak_sec

    thr = peak_val * threshold_ratio
    end_idx = min(len(audio), int((peak_sec + max_tail_sec) * sr))
    i = peak_idx
    while i < end_idx:
        s1 = min(len(audio), i + hop)
        v = float(np.sqrt(np.mean(audio[i:s1] ** 2) + 1e-12))
        if v < thr:
            break
        i += hop
    return min(duration_sec, i / sr)


def refine_event_timing(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """
    Snap classifier window centers to acoustic onsets and fix gate spans.

    Impulsive: spectral-flux onset; short pre-roll + decay tail.
    Sustained: RMS peak; span capped for gating metadata.
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    duration_sec = len(audio) / sr

    radius = taxonomy.impulsive_onset_search_radius_sec
    half = taxonomy.impulsive_event_half_width_sec
    pre_roll = taxonomy.impulsive_pre_roll_sec
    max_tail = taxonomy.impulsive_decay_tail_sec
    decay_thr = taxonomy.impulsive_decay_threshold
    sustained_max = taxonomy.sustained_max_gate_sec

    refined: list[DetectedEvent] = []
    for ev in events:
        cat_cfg = taxonomy.categories.get(ev.category)
        impulsive = bool(cat_cfg and cat_cfg.impulsive)

        if impulsive:
            acoustic_peak = _find_impulsive_peak(
                audio, sr, ev.peak_sec, taxonomy, duration_sec=duration_sec
            )
        else:
            acoustic_peak = _find_acoustic_peak(audio, sr, ev.peak_sec, radius)

        if impulsive:
            peak = acoustic_peak
            start = max(0.0, peak - pre_roll)
            # Keep some post-onset content for algorithm context
            min_end = peak + half
            decay_end = _extend_decay_tail(
                audio,
                sr,
                peak,
                threshold_ratio=decay_thr,
                max_tail_sec=max_tail,
                duration_sec=duration_sec,
            )
            end = min(duration_sec, max(min_end, decay_end))
        else:
            peak = acoustic_peak
            # Keep classifier span; do NOT recenter around peak (that invents
            # rumble during quiet gaps between intermittent bursts).
            start = max(0.0, ev.start_sec)
            end = min(duration_sec, max(ev.end_sec, peak + 0.05))
            if end < start:
                end = min(duration_sec, start + 0.25)
            # Soft cap: trim the quieter tail, keep the onset side
            if end - start > sustained_max and sustained_max > 0:
                end = min(end, start + sustained_max)
            peak = min(max(peak, start), end)

        refined.append(
            DetectedEvent(
                category=ev.category,
                label=ev.label,
                start_sec=start,
                peak_sec=peak,
                end_sec=end,
                confidence=ev.confidence,
                context_token=ev.context_token,
                audio_score=ev.audio_score,
                video_score=ev.video_score,
                sources=list(ev.sources),
            )
        )
    return refined
