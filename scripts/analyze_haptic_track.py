"""
Predict on-device haptic density for a haptic WAV, without a phone.

Mirrors `WavHapticParser.kt` so the window intensities printed here are the ones
the Android app will actually play, and reports them next to a rule-based JSON
haptic map so algorithm output can be compared against a known-good reference.

Usage:
    python scripts/analyze_haptic_track.py out/algorithm_*.wav \
        --reference "sample/videoplayback_output_haptic_map[1].json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_INTENSITY_COMPAT = 1
MIN_INTENSITY_DEFAULT = 18


def _percentile(sorted_values: np.ndarray, fraction: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    index = int(round((sorted_values.size - 1) * min(max(fraction, 0.0), 1.0)))
    return float(sorted_values[min(max(index, 0), sorted_values.size - 1)])


#: Fraction of the p10..p90 span used as the silence gate in legacy mode.
GATE_FRACTION_DEFAULT = 0.30
#: Absolute envelope below which a window is silent in direct mode.
DIRECT_NOISE_FLOOR = 0.02


def wav_to_track(
    path: Path,
    *,
    window_ms: int = 20,
    event_trigger_mode: bool = True,
    gate_fraction: float | None = None,
) -> tuple[list[int], int]:
    """
    Return per-window intensities (0-255) using the Android parser's math.

    Event-Trigger Mode maps the envelope straight to intensity, because the
    pipeline already shapes absolute levels (continuous layer low, event accents
    high). Legacy mode keeps the percentile stretch used for tracks that carry
    no continuous content of their own.
    """
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    samples_per_window = max(1, (sr * window_ms) // 1000)
    n_windows = int(np.ceil(audio.size / samples_per_window))
    padded = np.pad(audio, (0, n_windows * samples_per_window - audio.size))
    frames = padded.reshape(n_windows, samples_per_window)

    peak = np.max(np.abs(frames), axis=1)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    envelope = np.maximum(peak, rms * np.sqrt(2.0))

    signs = padded >= 0.0
    crossings = np.diff(signs.astype(np.int8)) != 0
    crossings_per_frame = np.pad(crossings, (1, 0)).reshape(n_windows, samples_per_window)
    zcr = crossings_per_frame.sum(axis=1) / samples_per_window

    if event_trigger_mode:
        track: list[int] = []
        for value in envelope:
            if value < DIRECT_NOISE_FLOOR:
                track.append(0)
                continue
            intensity = int(round(min(value, 1.0) * 255.0))
            track.append(0 if intensity < MIN_INTENSITY_COMPAT else min(intensity, 255))
        return track, window_ms

    radius = 8
    kernel = np.ones(2 * radius + 1)
    counts = np.convolve(np.ones(n_windows), kernel, mode="same")
    smoothed = np.convolve(envelope, kernel, mode="same") / counts
    envelope_for_combined = np.maximum(0.0, envelope - smoothed * 0.94)

    zcr_delta = np.abs(np.diff(zcr, prepend=zcr[0] if n_windows else 0.0))
    combined = np.maximum(0.0, 0.82 * envelope_for_combined + 0.18 * zcr_delta)

    sorted_combined = np.sort(combined)
    p_low = _percentile(sorted_combined, 0.50)
    p90 = max(_percentile(sorted_combined, 0.90), 1e-6)
    gate = p_low + (p90 - p_low) * (gate_fraction or GATE_FRACTION_DEFAULT)

    legacy: list[int] = []
    for value in combined:
        if value < gate:
            legacy.append(0)
            continue
        normalized = min(max((value - gate) / max(p90 - gate, 1e-6), 0.0), 1.0)
        intensity = int(round(normalized * 255.0))
        legacy.append(0 if intensity < MIN_INTENSITY_DEFAULT else min(intensity, 255))
    return legacy, window_ms


def json_to_track(path: Path) -> tuple[list[int], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["track"]
    window_ms = int(payload.get("window_size_ms", 40))
    keys = sorted(int(k) for k in raw)
    return [int(raw[str(k)]) for k in keys], window_ms


def summarize(name: str, track: list[int], window_ms: int) -> dict:
    values = np.array(track, dtype=int)
    active = values[values > 0]

    runs: list[int] = []
    current = 0
    for value in values:
        if value > 0:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)

    buckets = {
        "silent": int(np.sum(values == 0)),
        "1-31": int(np.sum((values >= 1) & (values < 32))),
        "32-63": int(np.sum((values >= 32) & (values < 64))),
        "64-127": int(np.sum((values >= 64) & (values < 128))),
        "128-200": int(np.sum((values >= 128) & (values <= 200))),
        "201-255": int(np.sum(values > 200)),
    }

    return {
        "name": name,
        "window_ms": window_ms,
        "windows": int(values.size),
        "active_pct": 100.0 * active.size / max(values.size, 1),
        "median_active": float(np.median(active)) if active.size else 0.0,
        "mean_active": float(np.mean(active)) if active.size else 0.0,
        "runs": len(runs),
        "longest_run_ms": (max(runs) * window_ms) if runs else 0,
        "buckets": buckets,
    }


def print_report(stats: list[dict]) -> None:
    header = f"{'track':<38}{'win':>5}{'active%':>9}{'med':>6}{'mean':>7}{'runs':>6}{'maxRun':>9}"
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['name'][:37]:<38}{s['window_ms']:>5}{s['active_pct']:>8.1f}%"
            f"{s['median_active']:>6.0f}{s['mean_active']:>7.1f}"
            f"{s['runs']:>6}{s['longest_run_ms'] / 1000:>8.1f}s"
        )

    print()
    print("intensity distribution (% of windows)")
    labels = ["silent", "1-31", "32-63", "64-127", "128-200", "201-255"]
    print(f"{'track':<38}" + "".join(f"{label:>10}" for label in labels))
    print("-" * (38 + 10 * len(labels)))
    for s in stats:
        total = max(s["windows"], 1)
        row = "".join(f"{100.0 * s['buckets'][label] / total:>9.1f}%" for label in labels)
        print(f"{s['name'][:37]:<38}{row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", nargs="*", type=Path, help="haptic WAV files to analyze")
    parser.add_argument("--reference", type=Path, help="rule-based JSON haptic map to compare to")
    parser.add_argument("--window-ms", type=int, default=20, help="WAV analysis window (Android default 20)")
    parser.add_argument(
        "--no-event-trigger",
        action="store_true",
        help="analyze with Event-Trigger Mode off (high-passed envelope, stricter gate)",
    )
    args = parser.parse_args()

    stats: list[dict] = []
    if args.reference:
        track, window_ms = json_to_track(args.reference)
        stats.append(summarize(f"[reference] {args.reference.name}", track, window_ms))

    for wav in args.wavs:
        track, window_ms = wav_to_track(
            wav,
            window_ms=args.window_ms,
            event_trigger_mode=not args.no_event_trigger,
        )
        stats.append(summarize(wav.name, track, window_ms))

    if not stats:
        parser.error("provide at least one WAV or a --reference JSON")
    print_report(stats)


if __name__ == "__main__":
    main()
