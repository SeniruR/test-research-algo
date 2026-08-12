"""Keep vehicle rumble where the clip is loud; drop quiet engine-bed FPs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.onset_refine import _envelope_rms
from haptic_gt.context.sustained_salience import apply_vehicle_salience
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def _has_energy_burst(
    times: np.ndarray,
    env: np.ndarray,
    *,
    center_sec: float,
    pre_sec: float,
    post_sec: float,
    rise_ratio: float,
) -> tuple[bool, float, float]:
    """
    Return (ok, onset_sec, end_sec) if energy rises vs a pre-window baseline.

    Steady engine bed fails; a clear rumble swell / burst passes.
    """
    if env.size == 0 or times.size == 0:
        return False, center_sec, center_sec

    pre_mask = (times >= center_sec - pre_sec) & (times < center_sec - 0.05)
    burst_mask = (times >= center_sec - 0.15) & (times <= center_sec + post_sec)
    if not np.any(burst_mask):
        return False, center_sec, center_sec

    burst_env = env[burst_mask]
    burst_times = times[burst_mask]
    peak_val = float(np.max(burst_env))
    if peak_val < 1e-8:
        return False, center_sec, center_sec

    if np.any(pre_mask):
        # Robust baseline: ignore short dips in the idle bed
        baseline = float(np.percentile(env[pre_mask], 60))
    else:
        baseline = float(np.percentile(burst_env, 25))

    baseline = max(baseline, peak_val * 1e-3)
    if peak_val < baseline * rise_ratio:
        return False, center_sec, center_sec

    # Onset: first frame that climbs well above baseline
    onset_thr = baseline + 0.35 * (peak_val - baseline)
    onset_idx = int(np.argmax(burst_env))  # fallback = peak
    for i, v in enumerate(burst_env):
        if v >= onset_thr:
            onset_idx = i
            break
    onset_sec = float(burst_times[onset_idx])

    # End: after peak, when energy falls near baseline again
    peak_i = int(np.argmax(burst_env))
    end_thr = baseline + 0.25 * (peak_val - baseline)
    end_idx = peak_i
    for i in range(peak_i, len(burst_env)):
        end_idx = i
        if burst_env[i] <= end_thr:
            break
    end_sec = float(burst_times[end_idx])
    if end_sec <= onset_sec:
        end_sec = onset_sec + 0.25
    return True, onset_sec, end_sec


def filter_sustained_rumble_bursts(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    report: dict | None = None,
) -> list[DetectedEvent]:
    """
    Gate vehicle rumble on absolute local RMS (salience), not RMS rise.

    Quiet high-rise bumps (engine bed / idle ticks) are dropped. Long loud
    plateaus AST missed are filled so auto-detect can catch marked rumbles
    on an unseen video. Impulsive events pass through unchanged.

    Pass ``report`` to record the loudness floor per scene and every proposal the
    gate dropped, so a missing rumble can be told apart from one the classifier
    never proposed.
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    hop_ms = taxonomy.onset_flux_hop_ms
    times, env = _envelope_rms(audio, sr, hop_ms=hop_ms)
    duration_sec = len(audio) / float(sr)
    return apply_vehicle_salience(
        events, times, env, taxonomy, duration_sec=duration_sec, report=report
    )
