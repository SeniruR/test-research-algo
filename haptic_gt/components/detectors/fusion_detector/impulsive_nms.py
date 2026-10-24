"""Keep one accent per blast, and keep accents from running into each other.

A cannon does not stop making spectral flux when it stops firing: measured on a
tank clip, the 0.9 s after the loudest shot holds dozens of local maxima at
0.3-0.65 of that shot's level, because the blast decay and its reverb ride over
the engine. None of them is a separate bang -- they rise only 1.0-2.0x above the
moment before them, where every real shot in the same clip rose 4.7-9.1x.

Two things then go wrong downstream:

1. Two events survive closer than ``impulsive_min_peak_distance_sec``, because
   the generic peak dedupe uses its own fixed margin, so one blast is reported
   as a shot plus a second shot somewhere in its own tail.
2. Refinement extends every span by its decay tail, so neighbouring accents
   overlap and the renderer hits the same bang twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .frozen_fusion import DetectedEvent
from .onset_refine import _spectral_flux, local_flux_ratio
from .taxonomy import Taxonomy, load_taxonomy

#: Accent timing is only meaningful to about this tolerance, so an attack is
#: looked for within it rather than exactly on the reported peak.
_ATTACK_WINDOW_SEC = 0.025


class _Attacks:
    """Spectral flux for one clip, queried at event peaks."""

    def __init__(self, source_wav: Path, taxonomy: Taxonomy) -> None:
        audio, sr = sf.read(source_wav, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.audio = audio.astype(np.float32)
        self.sr = int(sr)
        self.duration_sec = len(self.audio) / float(sr)
        self.times, self.flux = _spectral_flux(
            self.audio, self.sr, hop_ms=taxonomy.onset_flux_hop_ms
        )
        self.flux_peak = float(np.max(self.flux)) if self.flux.size else 0.0

    def _attack_index(self, t: float) -> int:
        """Strongest flux bin within the accent's own lead time.

        An attack can be a single 5 ms bin, so reading the bin nearest the peak
        reports less than half the real level when timing is off by one hop.
        """
        core = np.flatnonzero(np.abs(self.times - t) <= _ATTACK_WINDOW_SEC)
        if core.size == 0:
            return int(np.argmin(np.abs(self.times - t)))
        return int(core[int(np.argmax(self.flux[core]))])

    def level(self, t: float) -> float:
        if not self.flux.size or not self.times.size:
            return 0.0
        return float(self.flux[self._attack_index(t)])

    def rel_max(self, t: float) -> float:
        if self.flux_peak <= 0.0:
            return 0.0
        return self.level(t) / self.flux_peak

    def prominence(self, t: float) -> float:
        if not self.flux.size or not self.times.size:
            return 0.0
        return local_flux_ratio(
            self.times, self.flux, float(self.times[self._attack_index(t)])
        )


#: Decay bumps sit 0.2-0.4 s after a muzzle; this is the window in which there
#: is only one bang. Prefer the earliest sharp attack that is still loud enough
#: (muzzle), not the absolute loudest bin (often a boom 0.2-0.3 s later).
_SNAP_PROMINENCE = 3.0
_SNAP_EARLIEST_FRAC = 0.65


def _local_max_indices(flux: np.ndarray) -> np.ndarray:
    if flux.size < 3:
        return np.array([], dtype=int)
    return np.flatnonzero((flux[1:-1] >= flux[:-2]) & (flux[1:-1] >= flux[2:])) + 1


def snap_impulsive_peaks_to_attacks(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """Move each impulsive peak onto the muzzle in its own spacing window.

    Frame SED and the first refine pass often land 0.2-0.4 s late, on a bump in
    the blast's own decay. Promote then skips the real attack because it is
    inside the 0.40 s neighbour gate, so the haptic accent fires on the tail.
    Inside ``impulsive_min_peak_distance_sec`` there is only one bang: pick the
    earliest sharp attack that is still ≥65% of the loudest sharp peak in that
    window (prominence ≥ 3.0). Loudest-alone was landing on the boom after the
    muzzle and making the first haptic feel late.
    """
    taxonomy = taxonomy or load_taxonomy()
    if not any(_is_impulsive(e, taxonomy) for e in events):
        return list(events)

    attacks = _Attacks(Path(source_wav), taxonomy)
    if attacks.flux.size < 3:
        return list(events)

    radius = taxonomy.impulsive_min_peak_distance_sec
    pre_roll = taxonomy.impulsive_pre_roll_sec
    half = taxonomy.impulsive_event_half_width_sec
    peak_idxs = _local_max_indices(attacks.flux)

    out: list[DetectedEvent] = []
    for ev in events:
        if not _is_impulsive(ev, taxonomy):
            out.append(ev)
            continue
        best_t = _best_sharp_attack(
            attacks, peak_idxs, ev.peak_sec, radius=radius
        )
        if best_t is None or abs(best_t - ev.peak_sec) < 1e-4:
            out.append(ev)
            continue
        sources = list(ev.sources)
        if "flux_snap" not in sources:
            sources.append("flux_snap")
        start = max(0.0, best_t - pre_roll)
        end = min(attacks.duration_sec, max(ev.end_sec, best_t + half))
        out.append(
            DetectedEvent(
                category=ev.category,
                label=ev.label,
                start_sec=start,
                peak_sec=best_t,
                end_sec=max(end, start + 0.2),
                confidence=ev.confidence,
                context_token=ev.context_token,
                audio_score=ev.audio_score,
                video_score=ev.video_score,
                sources=sources,
                attack_rel_max=ev.attack_rel_max,
                attack_prominence=ev.attack_prominence,
            )
        )
    return out


def _best_sharp_attack(
    attacks: _Attacks,
    peak_idxs: np.ndarray,
    t: float,
    *,
    radius: float,
) -> float | None:
    """Earliest adequate sharp local maximum within ``radius`` of ``t``."""
    if peak_idxs.size == 0:
        return None
    nearby = peak_idxs[np.abs(attacks.times[peak_idxs] - t) <= radius]
    if nearby.size == 0:
        return None
    scored: list[tuple[float, float, float]] = []
    for idx in nearby:
        peak_t = float(attacks.times[idx])
        promin = local_flux_ratio(attacks.times, attacks.flux, peak_t)
        if promin < _SNAP_PROMINENCE:
            continue
        scored.append((float(attacks.flux[idx]), promin, peak_t))
    if not scored:
        return None
    loudest = max(row[0] for row in scored)
    floor = _SNAP_EARLIEST_FRAC * loudest
    adequate = [row for row in scored if row[0] >= floor]
    adequate.sort(key=lambda row: row[2])
    return adequate[0][2]


def _is_impulsive(ev: DetectedEvent, taxonomy: Taxonomy) -> bool:
    cfg = taxonomy.categories.get(ev.category)
    return bool(cfg is not None and cfg.impulsive)


def suppress_impulsive_overlaps(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """One accent per blast: enforce peak spacing, then stop spans overlapping.

    Ranking is by attack level rather than confidence, because frame posteriors
    from PANNs come out flat (every promoted shot reports the decode threshold),
    so confidence cannot say which of two neighbours is the real blast.
    """
    taxonomy = taxonomy or load_taxonomy()
    impulsive = [e for e in events if _is_impulsive(e, taxonomy)]
    if len(impulsive) < 2:
        return list(events)

    attacks = _Attacks(Path(source_wav), taxonomy)
    min_dist = taxonomy.impulsive_min_peak_distance_sec

    by_strength = sorted(
        impulsive,
        key=lambda e: (attacks.level(e.peak_sec), e.confidence),
        reverse=True,
    )
    kept: list[DetectedEvent] = []
    for ev in by_strength:
        if any(
            other.category == ev.category and abs(other.peak_sec - ev.peak_sec) < min_dist
            for other in kept
        ):
            continue
        kept.append(ev)

    kept.sort(key=lambda e: e.peak_sec)
    for i in range(len(kept) - 1):
        cur, nxt = kept[i], kept[i + 1]
        if cur.end_sec > nxt.start_sec:
            kept[i] = DetectedEvent(
                category=cur.category,
                label=cur.label,
                start_sec=cur.start_sec,
                peak_sec=cur.peak_sec,
                end_sec=max(nxt.start_sec, cur.peak_sec + 1e-3),
                confidence=cur.confidence,
                context_token=cur.context_token,
                audio_score=cur.audio_score,
                video_score=cur.video_score,
                sources=list(cur.sources),
                attack_rel_max=cur.attack_rel_max,
                attack_prominence=cur.attack_prominence,
            )

    out = [e for e in events if not _is_impulsive(e, taxonomy)]
    out += kept
    out.sort(key=lambda e: e.start_sec)
    return out


def split_rapid_fire(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """Add the shots of a burst that the one-bang-per-spacing rules merged away.

    Everything upstream keeps one bang per ``impulsive_min_peak_distance_sec``
    (0.45 s) so a cannon's boom and its decay bumps never become extra shots.
    A rifle fired every 0.33 s loses every second shot to that. After each shot
    in ``rapid_fire_categories``, a later sharp attack before the next shot (and
    within ``rapid_fire_window_sec`` of the last shot found) counts as a shot of
    its own only if it looks like one by itself: it rises
    ``rapid_fire_prominence`` above the moment before it (decay bumps rise 1-2.9x,
    real shots 4x and more) and reaches ``rapid_fire_rel_level`` of the shot
    before it. Then consecutive accents are kept ``impulsive_accent_gap_sec``
    apart, so a burst is felt as separate pulses instead of one long buzz.
    """
    taxonomy = taxonomy or load_taxonomy()
    cats = set(taxonomy.rapid_fire_categories)
    if not cats or not any(e.category in cats for e in events):
        return list(events)

    attacks = _Attacks(Path(source_wav), taxonomy)
    if attacks.flux.size < 3:
        return list(events)
    local_max = attacks.times[_local_max_indices(attacks.flux)]
    min_gap = taxonomy.rapid_fire_min_gap_sec
    pre_roll = taxonomy.impulsive_pre_roll_sec
    half = taxonomy.impulsive_event_half_width_sec

    impulsive = sorted(
        (e for e in events if _is_impulsive(e, taxonomy)), key=lambda e: e.peak_sec
    )
    peaks = [e.peak_sec for e in impulsive]
    added: list[DetectedEvent] = []
    for i, ev in enumerate(impulsive):
        if ev.category not in cats:
            continue
        next_peak = peaks[i + 1] if i + 1 < len(peaks) else attacks.duration_sec + min_gap
        floor = taxonomy.rapid_fire_rel_level * attacks.level(ev.peak_sec)
        t = ev.peak_sec
        while True:
            limit = min(t + taxonomy.rapid_fire_window_sec, next_peak - min_gap)
            shot = next(
                (
                    float(pt)
                    for pt in local_max
                    if t + min_gap <= pt <= limit
                    and attacks.prominence(float(pt)) >= taxonomy.rapid_fire_prominence
                    and attacks.level(float(pt)) >= floor
                ),
                None,
            )
            if shot is None:
                break
            start = max(0.0, shot - pre_roll)
            added.append(
                DetectedEvent(
                    category=ev.category,
                    label=ev.label,
                    start_sec=start,
                    peak_sec=shot,
                    end_sec=min(attacks.duration_sec, max(shot + half, start + 0.2)),
                    confidence=ev.confidence,
                    audio_score=ev.audio_score,
                    sources=["audio", "flux", "rapid_fire"],
                )
            )
            t = shot

    out = list(events) + added
    burst = sorted((e for e in out if e.category in cats), key=lambda e: e.peak_sec)
    gap = taxonomy.impulsive_accent_gap_sec
    for cur, nxt in zip(burst, burst[1:]):
        if cur.end_sec > nxt.start_sec - gap:
            cur.end_sec = max(nxt.start_sec - gap, cur.peak_sec + 0.05)
    out.sort(key=lambda e: e.start_sec)
    return out


def measure_impulsive_attacks(
    events: list[DetectedEvent],
    source_wav: str | Path,
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """Record how strong and how sharp each accent's attack actually was.

    Frame posteriors are reported as the decode threshold for every promoted
    shot, which hides the difference between the loudest cannon and a distant
    impact. These two numbers are what the promotion thresholds are tuned on.
    """
    taxonomy = taxonomy or load_taxonomy()
    if not any(_is_impulsive(e, taxonomy) for e in events):
        return list(events)

    attacks = _Attacks(Path(source_wav), taxonomy)
    out: list[DetectedEvent] = []
    for ev in events:
        if not _is_impulsive(ev, taxonomy):
            out.append(ev)
            continue
        ev.attack_rel_max = round(attacks.rel_max(ev.peak_sec), 4)
        ev.attack_prominence = round(attacks.prominence(ev.peak_sec), 3)
        out.append(ev)
    return out
