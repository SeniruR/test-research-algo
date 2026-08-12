"""Calibrate vehicle rumble detection against hand-marked peaks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.context.onset_refine import _envelope_rms
from haptic_gt.context.sustained_salience import (
    curve_value,
    local_rms,
    salience_stats,
    salience_threshold_curve,
    scene_spans,
)
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


@dataclass
class ManualPeakReport:
    manual_peak_sec: float
    matched_event_id: str | None
    matched_peak_sec: float | None
    match_error_sec: float | None
    detected_confidence: float | None
    audio_score: float | None
    local_rms: float | None
    scene_threshold: float
    rms_rise_ratio: float | None
    would_pass_salience: bool
    would_pass_rise_1_25: bool
    would_pass_rise_1_35: bool
    would_pass_rise_1_50: bool
    would_pass_rise_1_70: bool
    status: str  # "hit" | "miss"


@dataclass
class DetectedPeakReport:
    event_id: str
    peak_sec: float
    start_sec: float
    end_sec: float
    confidence: float
    audio_score: float | None
    nearest_manual_sec: float | None
    distance_sec: float | None
    local_rms: float | None
    rms_rise_ratio: float | None
    label: str  # "tp" | "fp" | "unknown"


def _rms_rise_at(
    times: np.ndarray,
    env: np.ndarray,
    center_sec: float,
    taxonomy: Taxonomy,
) -> float | None:
    pre = taxonomy.sustained_burst_pre_sec
    post = taxonomy.sustained_burst_post_sec
    pre_mask = (times >= center_sec - pre) & (times < center_sec - 0.05)
    burst_mask = (times >= center_sec - 0.15) & (times <= center_sec + post)
    if not np.any(burst_mask):
        return None
    peak_val = float(np.max(env[burst_mask]))
    if peak_val < 1e-8:
        return None
    if np.any(pre_mask):
        baseline = float(np.percentile(env[pre_mask], 60))
    else:
        baseline = float(np.percentile(env[burst_mask], 25))
    baseline = max(baseline, peak_val * 1e-3)
    return peak_val / baseline


def calibrate_rumble_thresholds(
    source_wav: str | Path,
    manual_peaks_sec: list[float],
    events_json: str | Path | dict | None = None,
    *,
    match_tolerance_sec: float = 1.0,
    taxonomy: Taxonomy | None = None,
) -> dict:
    """
    Compare hand-marked rumble times to detected events + local RMS.

    A mark is a hit if any vehicle span covers it (or peak is within
    ``match_tolerance_sec``). Use ``local_rms`` / ``would_pass_salience``
    to check the loudness gate.
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    times, env = _envelope_rms(audio, sr, hop_ms=taxonomy.onset_flux_hop_ms)

    events: list[dict] = []
    if events_json is not None:
        if isinstance(events_json, dict):
            payload = events_json
        else:
            payload = json.loads(Path(events_json).read_text(encoding="utf-8"))
        events = [
            e
            for e in payload.get("events", [])
            if str(e.get("category", "")).lower() == "vehicle"
        ]

    stats = salience_stats(env, taxonomy)
    thr = float(stats["threshold"])
    # The detector gates on a floor per scene, so a single clip-wide number can
    # say a mark passes while the scene it lives in rejects it.
    thr_curve = salience_threshold_curve(times, env, taxonomy)
    scenes = scene_spans(times, env, taxonomy)
    peak_win = taxonomy.sustained_salience_peak_win_sec

    def thr_at(t_sec: float) -> float:
        return curve_value(times, thr_curve, t_sec)

    def _covers(ev: dict, mark: float) -> float | None:
        peak = float(ev.get("peak_sec", ev.get("start_sec", 0.0)))
        start = float(ev.get("start_sec", peak))
        end = float(ev.get("end_sec", peak))
        if start - 0.2 <= mark <= end + 0.2:
            return 0.0
        err = abs(peak - mark)
        if err <= match_tolerance_sec:
            return err
        return None

    manual_rows: list[ManualPeakReport] = []
    for m in sorted(float(x) for x in manual_peaks_sec):
        rise = _rms_rise_at(times, env, m, taxonomy)
        rms = local_rms(times, env, m, half_win_sec=peak_win)
        best_i = None
        best_err = None
        for i, ev in enumerate(events):
            err = _covers(ev, m)
            if err is None:
                continue
            if best_err is None or err < best_err:
                best_err = err
                best_i = i
        if best_i is not None and best_err is not None:
            ev = events[best_i]
            peak = float(ev.get("peak_sec", m))
            manual_rows.append(
                ManualPeakReport(
                    manual_peak_sec=m,
                    matched_event_id=str(ev.get("event_id")),
                    matched_peak_sec=peak,
                    match_error_sec=round(best_err, 3),
                    detected_confidence=float(ev.get("confidence", 0.0)),
                    audio_score=(
                        float(ev["audio_score"])
                        if ev.get("audio_score") is not None
                        else None
                    ),
                    local_rms=round(rms, 4),
                    scene_threshold=round(thr_at(m), 4),
                    rms_rise_ratio=None if rise is None else round(rise, 3),
                    would_pass_salience=bool(rms >= thr_at(m)),
                    would_pass_rise_1_25=bool(rise is not None and rise >= 1.25),
                    would_pass_rise_1_35=bool(rise is not None and rise >= 1.35),
                    would_pass_rise_1_50=bool(rise is not None and rise >= 1.50),
                    would_pass_rise_1_70=bool(rise is not None and rise >= 1.70),
                    status="hit",
                )
            )
        else:
            manual_rows.append(
                ManualPeakReport(
                    manual_peak_sec=m,
                    matched_event_id=None,
                    matched_peak_sec=None,
                    match_error_sec=None,
                    detected_confidence=None,
                    audio_score=None,
                    local_rms=round(rms, 4),
                    scene_threshold=round(thr_at(m), 4),
                    rms_rise_ratio=None if rise is None else round(rise, 3),
                    would_pass_salience=bool(rms >= thr_at(m)),
                    would_pass_rise_1_25=bool(rise is not None and rise >= 1.25),
                    would_pass_rise_1_35=bool(rise is not None and rise >= 1.35),
                    would_pass_rise_1_50=bool(rise is not None and rise >= 1.50),
                    would_pass_rise_1_70=bool(rise is not None and rise >= 1.70),
                    status="miss",
                )
            )

    detected_rows: list[DetectedPeakReport] = []
    manuals = [float(x) for x in manual_peaks_sec]
    for i, ev in enumerate(events):
        peak = float(ev.get("peak_sec", ev.get("start_sec", 0.0)))
        start = float(ev.get("start_sec", peak))
        end = float(ev.get("end_sec", peak))
        rise = _rms_rise_at(times, env, peak, taxonomy)
        rms = local_rms(times, env, peak, half_win_sec=peak_win)
        if manuals:
            nearest = min(manuals, key=lambda m: abs(m - peak))
            dist = abs(nearest - peak)
            covered = any(start - 0.2 <= m <= end + 0.2 for m in manuals)
            label = "tp" if covered or dist <= match_tolerance_sec else "fp"
        else:
            nearest, dist, label = None, None, "unknown"
        detected_rows.append(
            DetectedPeakReport(
                event_id=str(ev.get("event_id", f"event_{i+1:03d}")),
                peak_sec=peak,
                start_sec=start,
                end_sec=end,
                confidence=float(ev.get("confidence", 0.0)),
                audio_score=(
                    float(ev["audio_score"]) if ev.get("audio_score") is not None else None
                ),
                nearest_manual_sec=nearest,
                distance_sec=None if dist is None else round(dist, 3),
                local_rms=round(rms, 4),
                rms_rise_ratio=None if rise is None else round(rise, 3),
                label=label,
            )
        )

    hits = sum(1 for r in manual_rows if r.status == "hit")
    misses = sum(1 for r in manual_rows if r.status == "miss")
    fps = sum(1 for r in detected_rows if r.label == "fp")
    report = {
        "summary": {
            "manual_count": len(manual_rows),
            "hits": hits,
            "misses": misses,
            "false_positives": fps,
            "salience_threshold": round(thr, 4),
            "salience_unimodal": bool(stats["unimodal"]),
            "scenes": [
                {
                    "start_sec": round(lo, 2),
                    "end_sec": round(hi, 2),
                    "threshold": round(thr_at((lo + hi) / 2), 4),
                }
                for lo, hi in scenes
            ],
            "current_rise_ratio_setting": taxonomy.sustained_burst_rise_ratio,
            "sustained_encoder_threshold": taxonomy.sustained_encoder_threshold,
            "hint": (
                "Hits use span coverage (a 29–32s island hits 29, 30, 31, 31.9). "
                "would_pass_salience compares each mark to its own scene's floor, "
                "which is what the detector gates on; salience_threshold is the "
                "clip-wide number and only applies to single-scene clips."
            ),
        },
        "manual": [asdict(r) for r in manual_rows],
        "detected": [asdict(r) for r in detected_rows],
    }
    return report


@dataclass
class TimelineSample:
    t_sec: float
    local_rms: float
    rms_rise_ratio: float | None
    nearest_detected_peak_sec: float | None
    nearest_detected_dist_sec: float | None
    nearest_manual_dist_sec: float | None
    in_manual_window: bool
    scene_threshold: float
    above_salience: bool
    would_pass_1_10: bool
    would_pass_1_25: bool
    would_pass_1_35: bool


def scan_rumble_timeline(
    source_wav: str | Path,
    *,
    start_sec: float = 28.0,
    end_sec: float = 45.0,
    hop_sec: float = 0.1,
    events_json: str | Path | dict | None = None,
    manual_peaks_sec: list[float] | None = None,
    manual_window_sec: float = 0.4,
    taxonomy: Taxonomy | None = None,
) -> dict:
    """
    Dense timeline scan (default 100 ms) of local RMS + rise.

    Auto rumble follows ``above_salience`` (absolute loudness), not rise.
    ``hop_sec=0.1`` is 28.0, 28.1, … (set 0.01 for 10 ms).
    """
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    duration = len(audio) / sr
    times, env = _envelope_rms(audio, sr, hop_ms=taxonomy.onset_flux_hop_ms)

    detected_peaks: list[float] = []
    if events_json is not None:
        if isinstance(events_json, dict):
            payload = events_json
        else:
            payload = json.loads(Path(events_json).read_text(encoding="utf-8"))
        detected_peaks = [
            float(e.get("peak_sec", e.get("start_sec", 0.0)))
            for e in payload.get("events", [])
            if str(e.get("category", "")).lower() == "vehicle"
        ]

    stats = salience_stats(env, taxonomy)
    thr = float(stats["threshold"])
    thr_curve = salience_threshold_curve(times, env, taxonomy)
    scenes = scene_spans(times, env, taxonomy)
    manuals = [float(x) for x in (manual_peaks_sec or [])]
    t0 = max(0.0, float(start_sec))
    t1 = min(duration, float(end_sec))
    hop = max(0.01, float(hop_sec))
    samples: list[TimelineSample] = []
    t = t0
    while t <= t1 + 1e-9:
        rms_val = float(np.interp(t, times, env)) if env.size else 0.0
        rise = _rms_rise_at(times, env, t, taxonomy)
        near_det = None
        dist_det = None
        if detected_peaks:
            near_det = min(detected_peaks, key=lambda p: abs(p - t))
            dist_det = abs(near_det - t)
        dist_man = None
        in_man = False
        if manuals:
            nearest_m = min(manuals, key=lambda m: abs(m - t))
            dist_man = abs(nearest_m - t)
            in_man = dist_man <= manual_window_sec
        samples.append(
            TimelineSample(
                t_sec=round(t, 3),
                local_rms=round(rms_val, 6),
                rms_rise_ratio=None if rise is None else round(rise, 3),
                nearest_detected_peak_sec=near_det,
                nearest_detected_dist_sec=None if dist_det is None else round(dist_det, 3),
                nearest_manual_dist_sec=None if dist_man is None else round(dist_man, 3),
                in_manual_window=in_man,
                scene_threshold=round(curve_value(times, thr_curve, t), 4),
                above_salience=bool(rms_val >= curve_value(times, thr_curve, t)),
                would_pass_1_10=bool(rise is not None and rise >= 1.10),
                would_pass_1_25=bool(rise is not None and rise >= 1.25),
                would_pass_1_35=bool(rise is not None and rise >= 1.35),
            )
        )
        t += hop

    return {
        "summary": {
            "start_sec": t0,
            "end_sec": t1,
            "hop_sec": hop,
            "n_samples": len(samples),
            "n_in_manual_window": sum(1 for s in samples if s.in_manual_window),
            "salience_threshold": round(thr, 4),
            "salience_unimodal": bool(stats["unimodal"]),
            "scenes": [
                {
                    "start_sec": round(lo, 2),
                    "end_sec": round(hi, 2),
                    "threshold": round(curve_value(times, thr_curve, (lo + hi) / 2), 4),
                }
                for lo, hi in scenes
            ],
            "max_rise": max((s.rms_rise_ratio or 0.0) for s in samples) if samples else 0.0,
            "hint": (
                "Auto rumble follows above_salience (local RMS vs its own scene's "
                "floor), not rise. Green marks should sit in high-RMS regions."
            ),
        },
        "samples": [asdict(s) for s in samples],
    }


def format_timeline_scan(report: dict, *, only_manual_windows: bool = False) -> str:
    """Plain-text dense scan. Optionally only rows near manual peaks."""
    lines = [
        "=== Dense rumble scan ===",
        json.dumps(report["summary"], indent=2),
        "",
        f"{'t':>7} {'rms':>8} {'loud?':>5} {'rise':>6} {'man?':>4} {'d_man':>6} {'d_det':>6}",
    ]
    for s in report["samples"]:
        if only_manual_windows and not s["in_manual_window"]:
            continue
        rise = "-" if s["rms_rise_ratio"] is None else f"{s['rms_rise_ratio']:.2f}"
        dman = "-" if s["nearest_manual_dist_sec"] is None else f"{s['nearest_manual_dist_sec']:.2f}"
        ddet = "-" if s["nearest_detected_dist_sec"] is None else f"{s['nearest_detected_dist_sec']:.2f}"
        lines.append(
            f"{s['t_sec']:7.1f} {s['local_rms']:8.4f} "
            f"{'Y' if s.get('above_salience') else '-':>5} {rise:>6} "
            f"{'Y' if s['in_manual_window'] else '-':>4} {dman:>6} {ddet:>6}"
        )
    return "\n".join(lines)


def format_calibration_table(report: dict) -> str:
    """Plain-text table for Colab / terminal."""
    lines = [
        "=== Rumble threshold calibration ===",
        json.dumps(report["summary"], indent=2),
        "",
        "Manual peaks:",
        f"{'manual':>8} {'status':<6} {'rms':>7} {'floor':>7} {'loud?':>5} "
        f"{'rise':>6} {'conf':>6} {'err':>6}",
    ]
    for r in report["manual"]:
        rise = "-" if r["rms_rise_ratio"] is None else f"{r['rms_rise_ratio']:.2f}"
        conf = "-" if r["detected_confidence"] is None else f"{r['detected_confidence']:.2f}"
        err = "-" if r["match_error_sec"] is None else f"{r['match_error_sec']:.2f}"
        rms = "-" if r.get("local_rms") is None else f"{r['local_rms']:.3f}"
        floor = "-" if r.get("scene_threshold") is None else f"{r['scene_threshold']:.3f}"
        lines.append(
            f"{r['manual_peak_sec']:8.2f} {r['status']:<6} {rms:>7} {floor:>7} "
            f"{'Y' if r.get('would_pass_salience') else '-':>5} {rise:>6} "
            f"{conf:>6} {err:>6}"
        )
    lines.extend(
        [
            "",
            "Detected vehicle events:",
            f"{'id':<12} {'start':>8} {'peak':>8} {'end':>8} {'label':<4} "
            f"{'rms':>7} {'conf':>6}",
        ]
    )
    for r in report["detected"]:
        rms = "-" if r.get("local_rms") is None else f"{r['local_rms']:.3f}"
        lines.append(
            f"{r['event_id']:<12} {r.get('start_sec', r['peak_sec']):8.2f} "
            f"{r['peak_sec']:8.2f} {r.get('end_sec', r['peak_sec']):8.2f} "
            f"{r['label']:<4} {rms:>7} {r['confidence']:6.2f}"
        )
    return "\n".join(lines)
