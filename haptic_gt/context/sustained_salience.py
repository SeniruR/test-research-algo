"""Vehicle rumble from absolute loudness (local RMS), not RMS rise.

AST spans are only a category prior. Haptic windows are the loud RMS islands
so quiet gaps stay quiet (a 10 s AST slab must not vibrate through a dip).

The loudness floor is measured per scene, not per clip. One clip can hold a
distant tank drive and a close car rumble twice its level; a single clip-wide
floor is then above the drive and below the car's idle bed, which loses the
drive completely and merges every car burst into one long buzz.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def local_rms(
    times,
    env,
    center_sec: float,
    *,
    half_win_sec: float = 0.15,
) -> float:
    """Peak envelope in a short window around ``center_sec``."""
    if env.size == 0 or times.size == 0:
        return 0.0
    mask = (times >= center_sec - half_win_sec) & (times <= center_sec + half_win_sec)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - center_sec)))
        return float(env[idx])
    return float(np.max(env[mask]))


def _median_smooth(env: np.ndarray, k: int = 15) -> np.ndarray:
    """Median filter (~75 ms at 5 ms hop) so 5 ms spikes cannot bridge islands."""
    if env.size == 0 or k <= 1:
        return env
    return median_filter(np.asarray(env, dtype=np.float64), size=int(k) | 1, mode="nearest")


def _frame_dt(times, default: float = 0.005) -> float:
    if times is None or len(times) < 2:
        return default
    dt = float(np.median(np.diff(times)))
    return dt if np.isfinite(dt) and dt > 0 else default


def _smoothing_frames(times, default: int = 15) -> int:
    """Median window in frames, ~75 ms — longer than one cycle of a 40 Hz rumble."""
    if times is None or len(times) < 2:
        return default
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        return default
    return max(5, int(round(0.075 / dt)) | 1)


def _active_frames(values: np.ndarray, taxonomy: Taxonomy) -> np.ndarray:
    """Drop silence before taking percentiles.

    In a region that is half silence, p50 lands inside the rumble and the gate
    ends up above the very thing it should keep.
    """
    if values.size == 0:
        return values
    silence_floor = (
        taxonomy.sustained_salience_silence_frac * float(np.percentile(values, 99))
    )
    active = values[values >= silence_floor]
    return active if active.size >= 8 else values


def _runs(flags: np.ndarray, dt: float) -> list[float]:
    """Lengths in seconds of each run of True in ``flags``."""
    out: list[float] = []
    run = 0
    for on in flags:
        if on:
            run += 1
        elif run:
            out.append(run * dt)
            run = 0
    if run:
        out.append(run * dt)
    return out


def _is_rhythmic(series: np.ndarray, threshold: float, dt: float, taxonomy: Taxonomy) -> bool:
    """True when the scene alternates between loud bursts and quiet gaps.

    Only the stretch between the first and last loud frame counts: the silence
    before a rumble starts is not a gap in it.

    Both sides have to be substantial. Gaps must be long enough that the islands
    would not bridge them anyway, and bursts at least as long as the shortest
    rumble -- otherwise track clanks over a steady drive read as a rhythm and the
    drive gets chopped into the spaces between its own clanks.
    """
    if series.size == 0 or dt <= 0:
        return False
    loud = np.where(series >= threshold)[0]
    if loud.size < 2:
        return False
    inside = series[loud[0] : loud[-1] + 1]
    gaps = [g for g in _runs(inside < threshold, dt) if g >= taxonomy.sustained_salience_gap_sec]
    bursts = _runs(inside >= threshold, dt)
    if len(gaps) < taxonomy.sustained_rhythm_min_gaps or not bursts:
        return False
    return (
        float(np.median(gaps)) <= taxonomy.sustained_rhythm_gap_max_sec
        and float(np.median(bursts)) >= taxonomy.sustained_salience_min_sec
    )


def _region_threshold(
    series: np.ndarray,
    taxonomy: Taxonomy,
    *,
    dt: float = 0.005,
) -> dict[str, float | str]:
    """Loudness floor for one scene, from the shape and the rhythm of its frames.

    A scene with two loudness modes is split between them, so gaps between bursts
    stay quiet. A scene with one mode is kept whole -- unless its quiet stretches
    repeat, which is a rumble easing off and coming back rather than a steady one.

    Spread is p90/p50, not p99/p50: the top percentile is a handful of frames, so
    track clanks over a steady drive read as a second loudness mode and the drive
    gets chopped into the gaps between its own clanks.
    """
    active = _active_frames(series, taxonomy)
    if active.size == 0:
        return {"threshold": 0.0, "spread": 1.0, "p_mid": 0.0, "p_loud": 0.0, "mode": "keep"}

    p_mid = float(np.percentile(active, taxonomy.sustained_salience_mid_pct))
    p_loud = float(np.percentile(active, 90))
    p_quiet = float(np.percentile(active, 10))
    spread = p_loud / max(p_mid, 1e-12)

    # Measured from the quiet level, not the median: once bursts fill more than
    # half the scene the median sits inside them and a median-based split lands on
    # top of the bursts it is supposed to keep.
    split = p_quiet + taxonomy.sustained_salience_mix * (p_loud - p_quiet)
    # Keeping all of it: half the median is not low enough -- a cut to a wider
    # camera angle halves the level while the tank keeps rolling -- so sit under
    # the quietest sustained part, with a floor so the gate stays meaningful when
    # that part is barely above silence.
    keep = max(0.6 * p_quiet, 0.30 * p_mid)

    bimodal = spread >= taxonomy.sustained_salience_min_sep
    # A rhythm needs two levels to alternate between; without this check the ripple
    # of a steady idle crosses its own split threshold and reads as a rhythm.
    two_level = p_loud >= taxonomy.sustained_rhythm_level_ratio * max(p_quiet, 1e-12)
    rhythmic = two_level and _is_rhythmic(series, split, dt, taxonomy)

    mode = "split" if (bimodal or rhythmic) else "keep"
    return {
        "threshold": float(split if mode == "split" else keep),
        "spread": float(spread),
        "p_mid": p_mid,
        "p_loud": p_loud,
        "mode": mode,
    }


def scene_spans(times, env, taxonomy: Taxonomy) -> list[tuple[float, float]]:
    """Split a clip where its sustained level steps to a new value for seconds.

    Clips cut together from several sources hold one recording level per source.
    Each is its own scene and gets its own loudness floor. A brief loud passage is
    not a scene -- a rumble burst over an engine bed must keep being measured
    against that bed, not against itself.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        return [(float(times[0]), float(times[-1]))] if times.size else []

    env_s = _median_smooth(np.asarray(env, dtype=np.float64), _smoothing_frames(times))
    start, end = float(times[0]), float(times[-1])
    block = max(taxonomy.sustained_scene_block_sec, _frame_dt(times) * 4)
    min_sec = taxonomy.sustained_scene_min_sec
    if end - start < 2 * min_sec:
        return [(start, end)]

    edges = np.arange(start, end, block)
    # How loud each block gets, not its median: a scene can be dynamic inside, and
    # a rumble that eases off between bursts must not read as a new scene.
    levels = np.array(
        [
            float(np.percentile(env_s[m], 90)) if np.any(m := ((times >= a) & (times < a + block))) else 0.0
            for a in edges
        ]
    )
    context = max(1, int(round(min_sec / block)))
    stable = taxonomy.sustained_scene_stable_ratio

    def _steady(chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        return float(np.max(chunk)) <= stable * max(float(np.min(chunk)), 1e-9)

    candidates: list[tuple[float, float]] = []
    for i in range(1, len(levels)):
        left, right = levels[max(0, i - context) : i], levels[i : i + context]
        if left.size < context or right.size < context:
            continue
        # Both sides must hold their own level, otherwise the "step" is a passage
        # inside one scene -- a rumble burst over a bed, not a change of source.
        if not (_steady(left) and _steady(right)):
            continue
        hi, lo = float(np.median(left)), float(np.median(right))
        if hi < lo:
            hi, lo = lo, hi
        ratio = hi / max(lo, 1e-9)
        if ratio >= taxonomy.sustained_scene_level_ratio:
            candidates.append((ratio, float(edges[i])))

    cuts: list[float] = []
    for _ratio, t in sorted(candidates, reverse=True):
        pieces = sorted([start, end, t, *cuts])
        if all(b - a >= min_sec for a, b in zip(pieces, pieces[1:], strict=False)):
            cuts.append(t)
    cuts.sort()

    bounds = [start, *cuts, end]
    return list(zip(bounds, bounds[1:], strict=False))


def _gate_source(
    env_s: np.ndarray,
    times,
    exclude_peaks: list[float] | None,
    exclude_radius_sec: float,
) -> np.ndarray:
    """Frames the loudness floor is measured on: blasts masked out, NaNs dropped."""
    keep = np.isfinite(env_s)
    if times is not None and exclude_peaks:
        mask = np.ones(len(env_s), dtype=bool)
        for peak in exclude_peaks:
            mask &= (times < peak - exclude_radius_sec) | (times > peak + exclude_radius_sec)
        if np.any(mask & keep):
            keep &= mask
    return env_s[keep]


def salience_stats(
    env,
    taxonomy: Taxonomy,
    *,
    times=None,
    exclude_peaks: list[float] | None = None,
    exclude_radius_sec: float = 0.7,
) -> dict[str, float | bool]:
    """Clip-wide loudness floor. Unimodal clips (steady idle) are not split."""
    # Smooth first, with the same window the islands use: on a raw 5 ms envelope a
    # low-frequency rumble swings within each cycle, which fakes a wide spread.
    env_s = _median_smooth(np.asarray(env, dtype=np.float64), _smoothing_frames(times))
    v = env_s[np.isfinite(env_s)]
    if v.size == 0:
        return {
            "threshold": 0.0,
            "sep": 1.0,
            "unimodal": True,
            "p_mid": 0.0,
            "p_loud": 0.0,
        }
    p_mid_raw = float(np.percentile(v, taxonomy.sustained_salience_mid_pct))
    p_loud_raw = float(np.percentile(v, taxonomy.sustained_salience_loud_pct))

    gate_src = _gate_source(env_s, times, exclude_peaks, exclude_radius_sec)
    out = _region_threshold(gate_src, taxonomy, dt=_frame_dt(times))

    sep = p_loud_raw / max(p_mid_raw, 1e-8)
    return {
        "threshold": out["threshold"],
        "sep": float(sep),
        "sep_active": out["spread"],
        # Callers use this to leave the classifier's spans alone. A rumble whose
        # quiet stretches repeat has gaps to keep quiet, so it is not one mode even
        # when its percentiles say so.
        "unimodal": bool(sep < taxonomy.sustained_salience_min_sep)
        and out["mode"] == "keep",
        "p_mid": out["p_mid"],
        "p_loud": out["p_loud"],
    }


def salience_threshold_curve(
    times,
    env,
    taxonomy: Taxonomy,
    *,
    exclude_peaks: list[float] | None = None,
    exclude_radius_sec: float = 0.7,
) -> np.ndarray:
    """Per-frame loudness floor: one level per scene.

    A clip with a single scene gets a single floor, the clip-wide one. Clips cut
    together from several sources get one floor per source, so a distant tank
    drive is not measured against a car rumble recorded twice as close.
    """
    times = np.asarray(times, dtype=np.float64)
    env_s = _median_smooth(np.asarray(env, dtype=np.float64), _smoothing_frames(times))
    if env_s.size == 0:
        return np.zeros(0, dtype=np.float64)

    global_thr = float(
        salience_stats(
            env,
            taxonomy,
            times=times,
            exclude_peaks=exclude_peaks,
            exclude_radius_sec=exclude_radius_sec,
        )["threshold"]
    )
    scenes = scene_spans(times, env, taxonomy)
    if len(scenes) <= 1:
        return np.full(env_s.size, global_thr, dtype=np.float64)

    keep = np.isfinite(env_s)
    if exclude_peaks:
        for peak in exclude_peaks:
            keep &= (times < peak - exclude_radius_sec) | (times > peak + exclude_radius_sec)
        if not np.any(keep):
            keep = np.isfinite(env_s)

    # A scene of near-silence has percentiles of near-silence, and its own floor
    # would gate room tone on as rumble. Nothing quieter than this is a vehicle.
    clip_floor = taxonomy.sustained_salience_scene_floor_frac * float(
        np.percentile(env_s[np.isfinite(env_s)], 99)
    )

    curve = np.full(env_s.size, global_thr, dtype=np.float64)
    for lo, hi in scenes:
        span = (times >= lo) & (times <= hi)
        frames = env_s[span & keep]
        if frames.size < 8:
            continue
        thr = float(_region_threshold(frames, taxonomy, dt=_frame_dt(times))["threshold"])
        curve[span] = max(thr, clip_floor)
    return curve


def rms_islands(
    times,
    env,
    threshold: float | np.ndarray,
    *,
    min_sec: float,
    gap_sec: float,
    min_duty: float = 0.55,
    edge_frac: float = 0.75,
    max_extend_sec: float = 0.4,
) -> list[tuple[float, float, float]]:
    """Contiguous high-RMS runs as (start, peak, end). Drops spiky low-duty blobs.

    ``threshold`` is a level or a per-frame floor (see
    :func:`salience_threshold_curve`).

    Edges are then walked outward down to ``edge_frac`` of the threshold: a rumble
    ramps up before it crosses the gate, and starting the buzz a third of a second
    into the burst feels late. The walk is capped so a gap cannot be swallowed.
    """
    if env.size == 0 or times.size == 0:
        return []
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.005
    dt = max(dt, 1e-4)
    smooth_k = max(5, int(round(0.075 / dt)) | 1)
    smooth = _median_smooth(env, k=smooth_k)
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(smooth.size, float(thr))
    mask = smooth >= thr
    gap_frames = max(1, int(round(gap_sec / dt)))
    min_frames = max(1, int(round(min_sec / dt)))

    closed = mask.copy()
    i = 0
    n = len(closed)
    while i < n:
        if closed[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= gap_frames:
            closed[i:j] = True
        i = j

    edge_thr = edge_frac * thr
    max_extend_frames = max(0, int(round(max_extend_sec / dt)))

    spans: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j < n and closed[j]:
            j += 1
        if (j - i) >= min_frames:
            sl = slice(i, j)
            duty = float(np.mean(mask[sl]))
            if duty >= min_duty:
                peak_i = i + int(np.argmax(env[sl]))
                lo = i
                limit = max(0, i - max_extend_frames)
                while lo > limit and smooth[lo - 1] >= edge_thr[lo - 1]:
                    lo -= 1
                hi = j - 1
                limit = min(n - 1, (j - 1) + max_extend_frames)
                while hi < limit and smooth[hi + 1] >= edge_thr[hi + 1]:
                    hi += 1
                spans.append((lo, peak_i, hi))
        i = j

    # Two bursts a short gap apart can each walk outward into the other; emit one
    merged: list[tuple[int, int, int]] = []
    for lo, peak_i, hi in spans:
        if merged and lo <= merged[-1][2]:
            p_lo, _, p_hi = merged[-1]
            hi = max(p_hi, hi)
            merged[-1] = (p_lo, p_lo + int(np.argmax(env[p_lo : hi + 1])), hi)
            continue
        merged.append((lo, peak_i, hi))

    return [
        (float(times[lo]), float(times[peak_i]), float(times[hi]))
        for lo, peak_i, hi in merged
    ]


def curve_value(times, values: np.ndarray, t_sec: float) -> float:
    """Value of a per-frame curve at ``t_sec``."""
    if values.size == 0:
        return 0.0
    return float(values[int(np.argmin(np.abs(np.asarray(times) - t_sec)))])


def crop_to_loud(
    times,
    env,
    center_sec: float,
    threshold: float,
    *,
    max_sec: float = 1.2,
    min_sec: float = 0.2,
) -> tuple[float, float, float]:
    """Tight start/peak/end around a loud peak; never keep a multi-second AST slab."""
    if env.size == 0 or times.size == 0:
        return center_sec, center_sec, center_sec + min_sec
    idx = int(np.argmin(np.abs(times - center_sec)))
    n = len(env)
    lo = idx
    while lo > 0 and env[lo - 1] >= threshold and (times[idx] - times[lo - 1]) <= max_sec / 2:
        lo -= 1
    hi = idx
    while hi + 1 < n and env[hi + 1] >= threshold and (times[hi + 1] - times[idx]) <= max_sec / 2:
        hi += 1
    start = float(times[lo])
    end = float(times[hi])
    if end - start < min_sec:
        pad = (min_sec - (end - start)) / 2
        start -= pad
        end += pad
    peak_i = lo + int(np.argmax(env[lo : hi + 1]))
    return start, float(times[peak_i]), end


def _overlaps(start: float, end: float, island: tuple[float, float, float], pad: float) -> bool:
    return start - pad <= island[2] and end + pad >= island[0]


def apply_vehicle_salience(
    events: list[DetectedEvent],
    times,
    env,
    taxonomy: Taxonomy | None = None,
    *,
    duration_sec: float | None = None,
    report: dict | None = None,
) -> list[DetectedEvent]:
    """
    Replace vehicle AST spans with loud RMS islands.

    One long classifier event may cover several rumbles; each island becomes
    its own haptic window. Quiet gaps are not filled. Short loud onsets with
    no island are cropped to ~1 s, never the original AST start/end.

    Pass ``report`` to record what the gate decided (floor per scene, which
    proposals it dropped and how far below the floor they were). Without it a
    missing rumble is indistinguishable from one the classifier never proposed.
    """
    taxonomy = taxonomy or load_taxonomy()
    passthrough: list[DetectedEvent] = []
    vehicles: list[DetectedEvent] = []
    for ev in events:
        if ev.category == "vehicle":
            vehicles.append(ev)
        else:
            passthrough.append(ev)

    if report is not None:
        report.update({"proposals": len(vehicles), "kept": 0, "dropped": []})

    if not vehicles:
        return list(events)

    imp_peaks = [
        e.peak_sec
        for e in passthrough
        if (c := taxonomy.categories.get(e.category)) is not None and c.impulsive
    ]
    stats = salience_stats(
        env, taxonomy, times=times, exclude_peaks=imp_peaks, exclude_radius_sec=0.7
    )
    thr_curve = salience_threshold_curve(
        times, env, taxonomy, exclude_peaks=imp_peaks, exclude_radius_sec=0.7
    )
    peak_win = taxonomy.sustained_salience_peak_win_sec

    if report is not None:
        scenes = scene_spans(times, env, taxonomy)
        report.update(
            {
                "clip_threshold": round(float(stats["threshold"]), 4),
                "clip_unimodal": bool(stats["unimodal"]),
                "scenes": [
                    {
                        "start_sec": round(lo, 2),
                        "end_sec": round(hi, 2),
                        "threshold": round(curve_value(times, thr_curve, (lo + hi) / 2), 4),
                    }
                    for lo, hi in scenes
                ],
            }
        )

    if stats["unimodal"]:
        if report is not None:
            report["kept"] = len(vehicles)
        return list(events)

    islands = rms_islands(
        times,
        env,
        thr_curve,
        min_sec=taxonomy.sustained_salience_min_sec,
        gap_sec=taxonomy.sustained_salience_gap_sec,
        min_duty=taxonomy.sustained_salience_min_duty,
        edge_frac=taxonomy.sustained_island_edge_frac,
        max_extend_sec=taxonomy.sustained_island_max_extend_sec,
    )

    kept: list[DetectedEvent] = list(passthrough)
    template = max(vehicles, key=lambda e: e.confidence)

    def _emit(ev: DetectedEvent, start: float, peak: float, end: float, sources: list[str]):
        if duration_sec is not None:
            end = min(end, duration_sec)
            start = min(max(0.0, start), end)
        kept.append(
            DetectedEvent(
                category="vehicle",
                label=ev.label or "Vehicle",
                start_sec=max(0.0, start),
                peak_sec=min(max(peak, start), max(end, start + 0.2)),
                end_sec=max(end, start + 0.2),
                confidence=ev.confidence,
                context_token=ev.context_token,
                audio_score=ev.audio_score,
                video_score=ev.video_score,
                sources=sources,
            )
        )

    for i, isl in enumerate(islands):
        # Skip only if this island's peak IS the blast, not the next volley ~0.5 s later
        if any(abs(isl[1] - p) <= 0.45 for p in imp_peaks):
            continue
        # A short island right after a blast is its decay, not engine rumble
        if isl[2] - isl[0] < 1.5 and any(
            isl[0] - 0.7 <= p <= isl[2] for p in imp_peaks
        ):
            continue
        best: DetectedEvent | None = None
        for ev in vehicles:
            if _overlaps(ev.start_sec, ev.end_sec, isl, 0.25) or (
                isl[0] - 0.15 <= ev.peak_sec <= isl[2] + 0.15
            ):
                if best is None or ev.confidence > best.confidence:
                    best = ev
        src_ev = best or template
        src = list(src_ev.sources) if src_ev.sources else ["audio"]
        if best is None and "salience" not in src:
            src = src + ["salience"]
        start, peak, end = isl
        _emit(src_ev, start, peak, end, src)

    max_orphan = taxonomy.sustained_salience_max_orphan_sec
    for ev in vehicles:
        if any(
            _overlaps(ev.start_sec, ev.end_sec, isl, 0.15)
            or (isl[0] - 0.15 <= ev.peak_sec <= isl[2] + 0.15)
            for isl in islands
        ):
            continue
        thr = curve_value(times, thr_curve, ev.peak_sec)
        loudness = local_rms(times, env, ev.peak_sec, half_win_sec=peak_win)
        if loudness < thr:
            if report is not None:
                report["dropped"].append(
                    {
                        "start_sec": round(ev.start_sec, 3),
                        "end_sec": round(ev.end_sec, 3),
                        "local_rms": round(loudness, 4),
                        "scene_threshold": round(thr, 4),
                        "reason": "below scene loudness floor",
                    }
                )
            continue
        start, peak, end = crop_to_loud(
            times, env, ev.peak_sec, thr, max_sec=max_orphan, min_sec=0.0
        )
        if end - start < taxonomy.sustained_salience_min_sec:
            if report is not None:
                report["dropped"].append(
                    {
                        "start_sec": round(ev.start_sec, 3),
                        "end_sec": round(ev.end_sec, 3),
                        "local_rms": round(loudness, 4),
                        "scene_threshold": round(thr, 4),
                        "reason": "loud span shorter than the minimum rumble",
                    }
                )
            continue
        src = list(ev.sources) if ev.sources else ["audio"]
        _emit(ev, start, peak, end, src)

    kept.sort(key=lambda e: e.start_sec)
    if report is not None:
        report["kept"] = sum(1 for e in kept if e.category == "vehicle")
    return kept
