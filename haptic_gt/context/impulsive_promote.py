"""Recover cannon/gunshot transients that AST labeled as vehicle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.onset_refine import _spectral_flux, local_flux_ratio
from haptic_gt.context.proposals import _local_peak_indices
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy, match_label_to_category

# AST confidence that marks an event as a real shot for level calibration
_CONFIRMED_CONF = 0.55
# When no confirmed shot exists, treat this fraction of clip max flux as the level
_FALLBACK_REF = 0.35
# Nearby AST explosion/gunshot score required to promote
_PROMOTE_AST_FLOOR = 0.20
_AST_REF_FRAC = 0.22
_AST_PROMINENCE = 2.2
# No AST support: must look like the confirmed shots
_FLUX_ONLY_REF_FRAC = 0.60
_FLUX_ONLY_PROMINENCE = 4.0
# Demotion: a fused "explosion" this far below shot level is rumble
_DEMOTE_REF_FRAC = 0.18

# An "anchor" blast, recognised by its own signature rather than by a model score.
# Confidence scales differ per backend (AST window scores run high, PANNs frame
# posteriors low), so a fixed 0.55 silently fails on some backends and leaves the
# shot level guessed from clip max -- which puts track clanks near "shot loud".
_ANCHOR_REL_MAX = 0.50
_ANCHOR_PROMINENCE = 4.0
# Inside a volley the decay of the previous blast inflates the pre-attack
# baseline, so prominence collapses on shots that are plainly loud. Near an
# anchor, lean on absolute level instead.
_VOLLEY_WINDOW_SEC = 2.0
_VOLLEY_REF_FRAC = 0.50
_VOLLEY_PROMINENCE = 1.3


def _flux_at(times: np.ndarray, flux: np.ndarray, peak_t: float) -> float:
    if flux.size == 0 or times.size == 0:
        return 0.0
    idx = int(np.argmin(np.abs(times - peak_t)))
    return float(flux[idx])


def _is_sharp_transient(
    times: np.ndarray,
    flux: np.ndarray,
    peak_t: float,
    flux_peak: float,
    *,
    ref_flux: float | None = None,
    local_ratio: float = 2.0,
) -> bool:
    """True blast: near shot level with a sharp attack. Rumble is neither."""
    if flux_peak < 1e-12:
        return False
    ref = ref_flux if ref_flux and ref_flux > 1e-12 else flux_peak * _FALLBACK_REF
    v = _flux_at(times, flux, peak_t)
    if v < _DEMOTE_REF_FRAC * ref:
        return False
    return local_flux_ratio(times, flux, peak_t) >= local_ratio


def _best_impulsive_score(
    scores: list[EncoderScore],
    taxonomy: Taxonomy,
    peak_t: float,
    win_sec: float,
) -> tuple[str, str, float] | None:
    nearby = [s for s in scores if abs(s.time_sec - peak_t) <= win_sec]
    best: tuple[str, str, float] | None = None
    for s in nearby:
        cat = match_label_to_category(taxonomy, s.label, s.source)
        if cat is None:
            continue
        cfg = taxonomy.categories.get(cat)
        if cfg is None or not cfg.impulsive:
            continue
        if best is None or s.score > best[2]:
            best = (cat, s.label, float(s.score))
    return best


def promote_impulsive_transients(
    events: list[DetectedEvent],
    source_wav: str | Path,
    encoder_scores: list[EncoderScore],
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """
    If a sharp flux peak has explosion/gunshot evidence, emit an impulsive
    event even when vehicle scored higher (engine bed under the muzzle).

    Quieter volley shots are found by local flux, not 32% of the loudest bang.
    Rumble-like AST explosions (tank drive) are dropped.
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    duration = len(audio) / float(sr)

    times, flux = _spectral_flux(audio, sr, hop_ms=taxonomy.onset_flux_hop_ms)
    if flux.size == 0:
        return list(events)

    flux_peak = float(np.max(flux))
    if flux_peak < 1e-12:
        return list(events)

    hop_sec = taxonomy.onset_flux_hop_ms / 1000.0
    min_dist = taxonomy.impulsive_min_peak_distance_sec
    min_frames = max(1, int(round(min_dist / hop_sec)))
    floor = flux_peak * 0.06
    idxs = _local_peak_indices(flux, min_score=floor, min_distance_frames=min_frames)

    clip_cat = "explosion"
    clip_label = "Explosion"
    clip_imp_score = 0.0
    for s in encoder_scores:
        cat = match_label_to_category(taxonomy, s.label, s.source)
        if cat is None:
            continue
        cfg = taxonomy.categories.get(cat)
        if cfg is None or not cfg.impulsive:
            continue
        if s.score > clip_imp_score:
            clip_imp_score = float(s.score)
            clip_cat = cat
            clip_label = s.label

    def _impulsive(ev: DetectedEvent) -> bool:
        cfg = taxonomy.categories.get(ev.category)
        return bool(cfg is not None and cfg.impulsive)

    # Calibrate "how loud is a shot in THIS clip". Track clanks are a small
    # fraction of that level; volley shots are not.
    ref_levels = [
        _flux_at(times, flux, e.peak_sec)
        for e in events
        if _impulsive(e) and e.confidence >= _CONFIRMED_CONF
    ]
    ref_levels = [v for v in ref_levels if v > 0.0]

    # Failing that, anchor on peaks that look like blasts on their own terms.
    anchors = [
        float(times[i])
        for i in idxs
        if float(flux[i]) >= _ANCHOR_REL_MAX * flux_peak
        and local_flux_ratio(times, flux, float(times[i])) >= _ANCHOR_PROMINENCE
    ]
    if not ref_levels and anchors:
        ref_levels = [_flux_at(times, flux, t) for t in anchors]

    ref_confirmed = len(ref_levels) >= 1
    ref_flux = (
        float(np.median(ref_levels)) if ref_confirmed else flux_peak * _FALLBACK_REF
    )
    if ref_flux < 1e-12:
        ref_flux = flux_peak * _FALLBACK_REF

    # Drop rumble-like fused explosions (drive noise scored as blast)
    surviving: list[DetectedEvent] = []
    for ev in events:
        if _impulsive(ev) and not _is_sharp_transient(
            times, flux, ev.peak_sec, flux_peak, ref_flux=ref_flux
        ):
            continue
        surviving.append(ev)

    existing_imp = [e for e in surviving if _impulsive(e)]
    half = taxonomy.impulsive_event_half_width_sec
    new_events: list[DetectedEvent] = []

    for idx in idxs:
        peak_t = float(times[idx])
        if any(abs(e.peak_sec - peak_t) <= 0.40 for e in existing_imp + new_events):
            continue
        v = _flux_at(times, flux, peak_t)
        prominence = local_flux_ratio(times, flux, peak_t)
        best = _best_impulsive_score(encoder_scores, taxonomy, peak_t, 0.7)
        has_nearby = best is not None and best[2] >= _PROMOTE_AST_FLOOR
        # With AST backing a quieter volley shot is enough; without it the peak
        # must look like the confirmed shots, not like a track clank.
        in_volley = any(abs(peak_t - a) <= _VOLLEY_WINDOW_SEC for a in anchors)
        if in_volley and v >= _VOLLEY_REF_FRAC * ref_flux:
            ok = prominence >= _VOLLEY_PROMINENCE
        elif has_nearby:
            ok = v >= _AST_REF_FRAC * ref_flux and prominence >= _AST_PROMINENCE
        else:
            ok = (
                ref_confirmed
                and clip_imp_score >= 0.35
                and v >= _FLUX_ONLY_REF_FRAC * ref_flux
                and prominence >= _FLUX_ONLY_PROMINENCE
            )
        if not ok:
            continue
        if has_nearby:
            cat, label, score = best
        else:
            cat, label = clip_cat, clip_label
            score = max(0.30, clip_imp_score * 0.55)
        start = max(0.0, peak_t - taxonomy.impulsive_pre_roll_sec)
        end = min(duration, peak_t + half)
        new_events.append(
            DetectedEvent(
                category=cat,
                label=label,
                start_sec=start,
                peak_sec=peak_t,
                end_sec=max(end, start + 0.2),
                confidence=max(score, taxonomy.impulsive_encoder_threshold),
                context_token=False,
                audio_score=score,
                video_score=None,
                sources=["audio", "flux"],
            )
        )
        existing_imp.append(new_events[-1])

    if not new_events:
        return surviving

    kept: list[DetectedEvent] = list(new_events)
    for ev in surviving:
        if not _impulsive(ev):
            if any(abs(ev.peak_sec - imp.peak_sec) <= 0.35 for imp in new_events):
                continue
            if any(
                imp.start_sec - 0.1 <= ev.peak_sec <= imp.end_sec + 0.1
                for imp in new_events
            ):
                continue
        kept.append(ev)
    kept.sort(key=lambda e: e.start_sec)
    return kept
