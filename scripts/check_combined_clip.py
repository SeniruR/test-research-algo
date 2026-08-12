"""Render the combined-clip fixture through A-D and print what the haptic does.

Not a test: it prints numbers for a human to sanity-check the gate and the
rendering on a clip cut together from cannons, a distant tank drive and a close
car rumble. Run it after touching the salience gate or the accent placement.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from haptic_gt.algorithms import freq_shift, haptic_gen, percept, pitch_match  # noqa: E402
from haptic_gt.context.frozen_fusion import DetectedEvent  # noqa: E402
from haptic_gt.context.rumble_filter import filter_sustained_rumble_bursts  # noqa: E402
from haptic_gt.context.taxonomy import load_taxonomy  # noqa: E402
from haptic_gt.haptic_synthesis import stitch_algorithm_output  # noqa: E402
from test_pipeline import _combined_scenes_audio  # noqa: E402

ALGORITHMS = {
    "A perception": percept.process_file,
    "B freq shift": freq_shift.process_file,
    "C pitch match": pitch_match.process_file,
    "D haptic gen": haptic_gen.process_file,
}


def _rms(audio: np.ndarray, sr: int, a: float, b: float) -> float:
    seg = audio[int(a * sr) : int(b * sr)]
    return float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0


def main() -> None:
    tax = load_taxonomy()
    sr = 22050
    audio, shots, drive, bursts = _combined_scenes_audio(sr)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav = td_path / "combined.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")

        proposals = [
            DetectedEvent("explosion", "Explosion", s - 0.08, s, s + 0.45, 0.4)
            for s in shots
        ]
        proposals += [
            DetectedEvent("vehicle", "Vehicle", drive[0], 13.0, drive[1], 0.6),
            DetectedEvent("vehicle", "Vehicle", 18.4, 22.0, 29.0, 0.7),
        ]
        report: dict = {}
        events = filter_sustained_rumble_bursts(proposals, wav, tax, report=report)

        print("scenes and their loudness floor")
        for scene in report["scenes"]:
            print(f"  {scene['start_sec']:6.2f}-{scene['end_sec']:6.2f}  {scene['threshold']:.4f}")
        print(f"clip-wide floor {report['clip_threshold']:.4f}  dropped {len(report['dropped'])}")

        print("\nrumble spans kept")
        for ev in events:
            if ev.category == "vehicle":
                print(f"  {ev.start_sec:6.2f}-{ev.end_sec:6.2f}  ({ev.end_sec - ev.start_sec:.2f}s)")
        covered = sum(
            max(0.0, min(e.end_sec, drive[1]) - max(e.start_sec, drive[0]))
            for e in events
            if e.category == "vehicle"
        )
        print(f"drive coverage {covered / (drive[1] - drive[0]):.0%}")

        print("\nrendered levels (RMS)")
        header = f"{'algorithm':14} {'drive':>8} {'car burst':>10} {'car gap':>8} {'shot':>8}"
        print(header)
        for name, fn in ALGORITHMS.items():
            out = td_path / f"{name[0]}.wav"
            stitch_algorithm_output(wav, events, out, fn)
            haptic, out_sr = sf.read(out)
            drv = _rms(haptic, out_sr, 12.0, 16.0)
            burst = _rms(haptic, out_sr, bursts[0][0] + 0.3, bursts[0][1] - 0.3)
            gap = _rms(haptic, out_sr, bursts[0][1] + 0.2, bursts[1][0] - 0.2)
            shot = _rms(haptic, out_sr, shots[0] - 0.02, shots[0] + 0.20)
            print(f"{name:14} {drv:8.4f} {burst:10.4f} {gap:8.4f} {shot:8.4f}")


if __name__ == "__main__":
    main()
