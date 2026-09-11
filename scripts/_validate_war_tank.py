"""Replay the impulsive stages on the real tank clip and check what users feel.

Seeds are the peaks the frame-SED run produced in Colab, so the promotion and
overlap rules can be checked without a GPU. Encoder scores are left empty on
purpose: that is the hardest case, where a blast has to be recovered from its
attack alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haptic_gt.context.frozen_fusion import DetectedEvent, dedupe_events_by_peak
from haptic_gt.context.impulsive_nms import (
    measure_impulsive_attacks,
    suppress_impulsive_overlaps,
)
from haptic_gt.context.impulsive_promote import promote_impulsive_transients
from haptic_gt.context.onset_refine import refine_event_timing
from haptic_gt.context.taxonomy import load_taxonomy

SRC = ROOT / "sample" / "_validate_war_tank" / "source_audio.wav"

# What the frame-SED run reported. 11.91 is a bump inside the 11.52 blast's own
# decay, not a second shot, so it must not survive as its own accent.
SEED_SHOTS = [
    (10.043, 10.123, 10.811),
    (10.921, 11.001, 11.455),
    (11.441, 11.521, 12.371),
    (11.831, 11.911, 12.361),
    (13.321, 13.401, 14.261),
    (14.931, 15.011, 15.861),
    (15.981, 16.061, 16.911),
]
# The second fireball, visible in the frame at 17.5 s, that the run never fired.
EXPECTED_EXTRA = [18.08]
# One accent per blast: these are decay bumps, not separate bangs.
MUST_COLLAPSE = [11.911]
# Track clanks during the drive: these must stay out of the haptic.
MUST_STAY_OUT = [1.94, 2.52, 3.46, 3.94, 9.60]


def main() -> int:
    tax = load_taxonomy()
    seeds = [
        DetectedEvent("explosion", "Explosion", s, p, e, 0.30, sources=["audio", "sed"])
        for s, p, e in SEED_SHOTS
    ]

    events = promote_impulsive_transients(seeds, SRC, [], tax)
    events = refine_event_timing(events, SRC, tax, relocate_impulsive_peaks=False)
    events = dedupe_events_by_peak(events)
    events = suppress_impulsive_overlaps(events, SRC, tax)
    events = measure_impulsive_attacks(events, SRC, tax)

    shots = sorted(
        (e for e in events if e.category in ("explosion", "gunshot")),
        key=lambda e: e.peak_sec,
    )
    peaks = [e.peak_sec for e in shots]

    print(f"{'start':>7} {'peak':>7} {'end':>7} {'rel':>6} {'promin':>7}")
    for ev in shots:
        print(
            f"{ev.start_sec:7.2f} {ev.peak_sec:7.2f} {ev.end_sec:7.2f} "
            f"{ev.attack_rel_max:6.3f} {ev.attack_prominence:7.2f}"
        )

    failures: list[str] = []
    print("\nchecks")

    overlaps = [
        (round(a.peak_sec, 2), round(b.peak_sec, 2))
        for a, b in zip(shots, shots[1:])
        if a.end_sec > b.start_sec + 1e-6
    ]
    print(f"  overlapping accent spans: {len(overlaps)}")
    if overlaps:
        failures.append(f"spans overlap: {overlaps}")

    too_close = [
        (round(a, 2), round(b, 2))
        for a, b in zip(peaks, peaks[1:])
        if b - a < tax.impulsive_min_peak_distance_sec - 1e-6
    ]
    print(f"  peaks closer than {tax.impulsive_min_peak_distance_sec}s: {len(too_close)}")
    if too_close:
        failures.append(f"peaks too close: {too_close}")

    for t in EXPECTED_EXTRA:
        hit = any(abs(p - t) <= 0.35 for p in peaks)
        print(f"  recovered blast at {t:5.2f}s: {'yes' if hit else 'NO'}")
        if not hit:
            failures.append(f"missed blast at {t}s")

    for t in MUST_COLLAPSE:
        bad = any(abs(p - t) <= 0.06 for p in peaks)
        print(f"  decay bump at {t:5.2f}s collapsed: {'NO' if bad else 'yes'}")
        if bad:
            failures.append(f"decay bump at {t}s still fires")

    for t in MUST_STAY_OUT:
        bad = any(abs(p - t) <= 0.30 for p in peaks)
        print(f"  drive clank at {t:5.2f}s kept out: {'NO' if bad else 'yes'}")
        if bad:
            failures.append(f"promoted drive clank at {t}s")

    weak = [
        round(e.peak_sec, 2)
        for e in shots
        if e.attack_prominence is not None and e.attack_prominence < 2.0
    ]
    print(f"  accents on rumble-like peaks: {len(weak)} {weak}")
    if weak:
        failures.append(f"accents not on sharp attacks: {weak}")

    reported = all(e.attack_rel_max is not None for e in shots)
    print(f"  attack strength reported: {'yes' if reported else 'NO'}")
    if not reported:
        failures.append("attack strength missing")

    print(f"\nshots: {[round(p, 2) for p in peaks]}")
    if failures:
        print("\nFAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
