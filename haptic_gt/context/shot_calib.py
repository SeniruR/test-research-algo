"""Calibrate impulsive (cannon / gunshot) detection against hand-marked times.

Mirrors ``rumble_calib`` for blasts: for every marked shot it reports the
nearest spectral-flux attack, how loud that attack is relative to the clip's
confirmed shots, and whether an event matched. Use it to decide whether a
timing error is in the detector or in the mark.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from haptic_gt.context.onset_refine import _spectral_flux, local_flux_ratio
from haptic_gt.context.proposals import _local_peak_indices
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy

IMPULSIVE_CATEGORIES = ("explosion", "gunshot", "weather")
CONFIRMED_CONF = 0.55


@dataclass
class ManualShotReport:
    manual_sec: float
    matched_event_id: str | None
    matched_peak_sec: float | None
    match_error_sec: float | None
    detected_confidence: float | None
    nearest_attack_sec: float | None
    attack_offset_sec: float | None
    flux_rel_shot_level: float | None
    prominence: float | None
    status: str  # "hit" | "miss"


@dataclass
class DetectedShotReport:
    event_id: str
    peak_sec: float
    confidence: float
    audio_score: float | None
    nearest_manual_sec: float | None
    distance_sec: float | None
    flux_rel_shot_level: float | None
    prominence: float | None
    label: str  # "tp" | "fp"


def _load_events(events_json: str | Path | dict | None) -> list[dict]:
    if events_json is None:
        return []
    if isinstance(events_json, dict):
        payload = events_json
    else:
        payload = json.loads(Path(events_json).read_text(encoding="utf-8"))
    return [
        e
        for e in payload.get("events", [])
        if str(e.get("category", "")).lower() in IMPULSIVE_CATEGORIES
    ]


def _flux_profile(
    source_wav: Path, taxonomy: Taxonomy
) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = sf.read(source_wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return _spectral_flux(audio, sr, hop_ms=taxonomy.onset_flux_hop_ms)


def _flux_at(times: np.ndarray, flux: np.ndarray, t: float) -> float:
    if flux.size == 0:
        return 0.0
    return float(flux[int(np.argmin(np.abs(times - t)))])


def _shot_level(
    times: np.ndarray, flux: np.ndarray, events: list[dict]
) -> tuple[float, bool]:
    """Reference attack strength of this clip's confident blasts."""
    levels = [
        _flux_at(times, flux, float(e.get("peak_sec", 0.0)))
        for e in events
        if float(e.get("confidence", 0.0)) >= CONFIRMED_CONF
    ]
    levels = [v for v in levels if v > 0.0]
    if levels:
        return float(np.median(levels)), True
    peak = float(np.max(flux)) if flux.size else 0.0
    return peak * 0.35, False


def _nearest_attack(
    times: np.ndarray,
    flux: np.ndarray,
    peak_idxs: list[int],
    t: float,
) -> float | None:
    if not peak_idxs:
        return None
    cand = min(peak_idxs, key=lambda i: abs(float(times[i]) - t))
    return float(times[cand])


def calibrate_shot_times(
    source_wav: str | Path,
    manual_shots_sec: list[float],
    events_json: str | Path | dict | None = None,
    *,
    match_tolerance_sec: float = 0.35,
    taxonomy: Taxonomy | None = None,
) -> dict:
    """Compare hand-marked shot times to detected impulsive events + flux attacks."""
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    times, flux = _flux_profile(source_wav, taxonomy)
    events = _load_events(events_json)
    shot_level, confirmed = _shot_level(times, flux, events)

    hop_sec = taxonomy.onset_flux_hop_ms / 1000.0
    min_frames = max(1, int(round(taxonomy.impulsive_min_peak_distance_sec / hop_sec)))
    flux_peak = float(np.max(flux)) if flux.size else 0.0
    peak_idxs = _local_peak_indices(
        flux,
        min_score=flux_peak * 0.05,
        min_distance_frames=min_frames,
    )

    def _rel(t: float) -> float | None:
        if shot_level <= 0.0:
            return None
        return round(_flux_at(times, flux, t) / shot_level, 3)

    manual_rows: list[ManualShotReport] = []
    used: set[str] = set()
    for mark in sorted(manual_shots_sec):
        best: tuple[float, dict] | None = None
        for ev in events:
            err = abs(float(ev.get("peak_sec", 0.0)) - mark)
            if err <= match_tolerance_sec and (best is None or err < best[0]):
                best = (err, ev)
        attack = _nearest_attack(times, flux, peak_idxs, mark)
        row = ManualShotReport(
            manual_sec=round(mark, 3),
            matched_event_id=None,
            matched_peak_sec=None,
            match_error_sec=None,
            detected_confidence=None,
            nearest_attack_sec=None if attack is None else round(attack, 3),
            attack_offset_sec=None if attack is None else round(attack - mark, 3),
            flux_rel_shot_level=_rel(mark if attack is None else attack),
            prominence=round(
                local_flux_ratio(times, flux, mark if attack is None else attack), 2
            ),
            status="miss",
        )
        if best is not None:
            err, ev = best
            row.matched_event_id = str(ev.get("event_id"))
            row.matched_peak_sec = round(float(ev.get("peak_sec", 0.0)), 3)
            row.match_error_sec = round(err, 3)
            row.detected_confidence = float(ev.get("confidence", 0.0))
            row.status = "hit"
            used.add(row.matched_event_id)
        manual_rows.append(row)

    detected_rows: list[DetectedShotReport] = []
    for ev in events:
        peak = float(ev.get("peak_sec", 0.0))
        nearest = (
            min(manual_shots_sec, key=lambda m: abs(m - peak))
            if manual_shots_sec
            else None
        )
        eid = str(ev.get("event_id"))
        detected_rows.append(
            DetectedShotReport(
                event_id=eid,
                peak_sec=round(peak, 3),
                confidence=float(ev.get("confidence", 0.0)),
                audio_score=ev.get("audio_score"),
                nearest_manual_sec=None if nearest is None else round(nearest, 3),
                distance_sec=None if nearest is None else round(abs(nearest - peak), 3),
                flux_rel_shot_level=_rel(peak),
                prominence=round(local_flux_ratio(times, flux, peak), 2),
                label="tp" if eid in used else "fp",
            )
        )

    hits = sum(1 for r in manual_rows if r.status == "hit")
    return {
        "summary": {
            "manual_count": len(manual_rows),
            "hits": hits,
            "misses": len(manual_rows) - hits,
            "detected_count": len(detected_rows),
            "false_positives": sum(1 for r in detected_rows if r.label == "fp"),
            "shot_level_flux": round(shot_level, 4),
            "shot_level_from_events": confirmed,
            "match_tolerance_sec": match_tolerance_sec,
        },
        "manual_shots": [asdict(r) for r in manual_rows],
        "detected_shots": [asdict(r) for r in detected_rows],
    }


def scan_shot_attacks(
    source_wav: str | Path,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    top_n: int = 40,
    taxonomy: Taxonomy | None = None,
) -> dict:
    """List the strongest flux attacks in a range — the clip's actual transients."""
    taxonomy = taxonomy or load_taxonomy()
    source_wav = Path(source_wav)
    times, flux = _flux_profile(source_wav, taxonomy)
    if flux.size == 0:
        return {"summary": {"count": 0}, "attacks": []}

    hop_sec = taxonomy.onset_flux_hop_ms / 1000.0
    min_frames = max(1, int(round(taxonomy.impulsive_min_peak_distance_sec / hop_sec)))
    flux_peak = float(np.max(flux))
    idxs = _local_peak_indices(
        flux,
        min_score=flux_peak * 0.05,
        min_distance_frames=min_frames,
    )
    end_sec = float(times[-1]) if end_sec is None else end_sec

    rows = []
    for i in idxs:
        t = float(times[i])
        if t < start_sec or t > end_sec:
            continue
        rows.append(
            {
                "t_sec": round(t, 3),
                "flux": round(float(flux[i]), 4),
                "flux_rel_max": round(float(flux[i]) / flux_peak, 3),
                "prominence": round(local_flux_ratio(times, flux, t), 2),
            }
        )
    rows.sort(key=lambda r: r["flux"], reverse=True)
    rows = rows[:top_n]
    rows.sort(key=lambda r: r["t_sec"])
    return {
        "summary": {
            "count": len(rows),
            "flux_max": round(flux_peak, 4),
            "start_sec": start_sec,
            "end_sec": round(end_sec, 3),
        },
        "attacks": rows,
    }


def _num(value, digits: int = 3, width: int = 8, signed: bool = False) -> str:
    """Fixed-width number or dash. Kept quote-free for Python 3.10 f-strings."""
    if value is None:
        return "-".rjust(width)
    sign = "+" if signed else ""
    return format(float(value), sign + "." + str(digits) + "f").rjust(width)


def format_shot_calibration_table(report: dict) -> str:
    s = report["summary"]
    level_src = (
        "from events"
        if s["shot_level_from_events"]
        else "(fallback: 35% of clip max)"
    )
    hits = s["hits"]
    total = s["manual_count"]
    lines = [
        "Shots: {0}/{1} hits, {2} misses, {3} false positives (tolerance {4}s)".format(
            hits, total, s["misses"], s["false_positives"], s["match_tolerance_sec"]
        ),
        "Shot level (flux of confident blasts): {0} {1}".format(
            s["shot_level_flux"], level_src
        ),
        "",
        "   mark status      event     peak     err   attack  d(att)  flux/shot  promin",
    ]
    for r in report["manual_shots"]:
        lines.append(
            _num(r["manual_sec"], 2, 7)
            + r["status"].rjust(7)
            + (r["matched_event_id"] or "-").rjust(11)
            + _num(r["matched_peak_sec"], 3, 9)
            + _num(r["match_error_sec"], 3, 8)
            + _num(r["nearest_attack_sec"], 3, 9)
            + _num(r["attack_offset_sec"], 3, 8, signed=True)
            + _num(r["flux_rel_shot_level"], 3, 11)
            + _num(r["prominence"], 2, 8)
        )

    lines += [
        "",
        "     event     peak   conf  label  nearest    dist  flux/shot  promin",
    ]
    for r in report["detected_shots"]:
        lines.append(
            r["event_id"].rjust(10)
            + _num(r["peak_sec"], 3, 9)
            + _num(r["confidence"], 2, 7)
            + r["label"].rjust(7)
            + _num(r["nearest_manual_sec"], 2, 9)
            + _num(r["distance_sec"], 3, 8)
            + _num(r["flux_rel_shot_level"], 3, 11)
            + _num(r["prominence"], 2, 8)
        )
    return "\n".join(lines)


def format_shot_scan(scan: dict) -> str:
    s = scan["summary"]
    lines = [
        "Strongest attacks {0}-{1}s (flux max {2}), {3} rows".format(
            s["start_sec"], s["end_sec"], s["flux_max"], s["count"]
        ),
        "   t_sec       flux  rel_max  promin",
    ]
    for r in scan["attacks"]:
        lines.append(
            _num(r["t_sec"], 3, 8)
            + _num(r["flux"], 4, 11)
            + _num(r["flux_rel_max"], 3, 9)
            + _num(r["prominence"], 2, 8)
        )
    return "\n".join(lines)
