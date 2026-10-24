"""Cut a full clip's pipeline outputs to match a trimmed copy of its video.

Running detection on a trimmed video gives different vibrations: thresholds are
set from the whole clip, and a re-encoded picture changes the flash scan and the
scene labels. Instead, run the pipeline once on the full video and cut its
outputs the same way the video was cut. The cuts are found by matching the
trimmed video's soundtrack against the full clip's, so the trimmed audio has to
be the original sound (no added music or re-mixed soundtrack).

    python scripts/trim_outputs.py FULL_RESULT_FOLDER TRIMMED_VIDEO OUT_FOLDER [--zip]

FULL_RESULT_FOLDER is the extracted ZIP of the full video's run (A-E WAVs at the
root, debug/context_detector/events.json, debug/audio_preparation/source_audio.wav).
OUT_FOLDER gets the trimmed video, the cut A-E WAVs and the matching events.json,
laid out like a pipeline ZIP so the app loads it the same way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haptic_gt.core.audio_io import extract_audio_from_video  # noqa: E402

ALIGN_SR = 8_000
FRAMES_PER_SEC = 100
BLOCK_SEC = 3.0
BLOCK_HOP_SEC = 0.5
#: A block counts as matched only at this normalised correlation of loudness envelopes.
MATCH_SCORE = 0.9
#: Consecutive matched blocks within this many seconds of each other share one cut.
SAME_OFFSET_SEC = 0.05
#: Below this share of the trimmed clip's blocks matched, the audio is not the original's.
#: The 156 s Fate of the Furious trim matched 70%; a re-mixed export of it, 1%.
MIN_COVERAGE = 0.5


@dataclass
class Segment:
    """Trimmed time [start, end) plays the full clip at start + offset."""

    start: float
    end: float
    offset: float
    score: float = 1.0


def envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """Log loudness every 10 ms, normalised; survives re-encoding and gain changes."""
    hop = sr // FRAMES_PER_SEC
    n = len(audio) // hop
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1))
    env = np.log(rms + 1e-4)
    return (env - env.mean()) / (env.std() + 1e-9)


def _ncc(block: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of ``block`` at every lag in ``full``."""
    m = len(block)
    b = block - block.mean()
    num = fftconvolve(full, b[::-1], mode="valid")
    c1 = np.concatenate([[0.0], np.cumsum(full)])
    c2 = np.concatenate([[0.0], np.cumsum(full**2)])
    s1 = c1[m:] - c1[:-m]
    var = (c2[m:] - c2[:-m]) - s1**2 / m
    score = num / (np.sqrt(np.maximum(var, 1e-9)) * np.linalg.norm(b) + 1e-9)
    score[var < 0.05 * m] = 0.0  # a flat (silent) stretch matches anything
    return score


def _matched_blocks(trim_env: np.ndarray, full_env: np.ndarray) -> list[tuple[float, float, float]]:
    """(trimmed time, offset, score) for every block that matches somewhere."""
    block = int(BLOCK_SEC * FRAMES_PER_SEC)
    hop = int(BLOCK_HOP_SEC * FRAMES_PER_SEC)
    out = []
    for s in range(0, max(len(trim_env) - block, 0) + 1, hop):
        c = _ncc(trim_env[s : s + block], full_env)
        if c.size == 0:
            continue
        k = int(np.argmax(c))
        if c[k] >= MATCH_SCORE:
            out.append((s / FRAMES_PER_SEC, (k - s) / FRAMES_PER_SEC, float(c[k])))
    return out


def _runs(blocks: list[tuple[float, float, float]]) -> list[list[tuple[float, float, float]]]:
    """Group matched blocks into runs that share one offset; lone stray blocks are dropped."""
    runs: list[list[tuple[float, float, float]]] = []
    for blk in blocks:
        if runs and abs(blk[1] - np.median([b[1] for b in runs[-1]])) <= SAME_OFFSET_SEC:
            runs[-1].append(blk)
        else:
            runs.append([blk])
    runs = [r for r in runs if len(r) >= 2]
    merged: list[list[tuple[float, float, float]]] = []
    for run in runs:
        if merged and abs(np.median([b[1] for b in merged[-1]]) - np.median([b[1] for b in run])) <= SAME_OFFSET_SEC:
            merged[-1].extend(run)  # same cut on both sides of a quiet stretch
        else:
            merged.append(run)
    return merged


def _cut_between(trim_env, full_env, lo: int, hi: int, off_a: int, off_b: int) -> int:
    """Frame in [lo, hi] where switching from offset a to offset b fits the audio best."""
    t = np.arange(lo, hi)
    ia, ib = t + off_a, t + off_b
    valid = (ia >= 0) & (ia < len(full_env)) & (ib >= 0) & (ib < len(full_env))
    if not valid.any():
        return (lo + hi) // 2
    err_a = np.where(valid, np.abs(trim_env[t] - full_env[np.clip(ia, 0, len(full_env) - 1)]), 0.0)
    err_b = np.where(valid, np.abs(trim_env[t] - full_env[np.clip(ib, 0, len(full_env) - 1)]), 0.0)
    total = np.concatenate([[0.0], np.cumsum(err_a)]) + (err_b.sum() - np.concatenate([[0.0], np.cumsum(err_b)]))
    return lo + int(np.argmin(total))


def _refine_offset(trim: np.ndarray, full: np.ndarray, seg: Segment) -> float:
    """Sharpen a 10 ms offset to the sample on the waveform of the segment's middle."""
    mid = 0.5 * (seg.start + seg.end)
    half = min(5.0, 0.5 * (seg.end - seg.start))
    a0, a1 = int((mid - half) * ALIGN_SR), int((mid + half) * ALIGN_SR)
    pad = int(0.05 * ALIGN_SR)
    b0 = int((mid - half + seg.offset) * ALIGN_SR) - pad
    b1 = int((mid + half + seg.offset) * ALIGN_SR) + pad
    if a1 - a0 < ALIGN_SR or b0 < 0 or b1 > len(full):
        return seg.offset
    a = trim[a0:a1] - trim[a0:a1].mean()
    c = fftconvolve(full[b0:b1], a[::-1], mode="valid")
    k = int(np.argmax(np.abs(c)))
    return (b0 + k - a0) / ALIGN_SR


def find_segments(trim: np.ndarray, full: np.ndarray, sr: int = ALIGN_SR) -> tuple[list[Segment], float]:
    """Which part of the full clip each stretch of the trimmed clip comes from.

    Also returns the share of the trimmed clip whose own blocks matched that cut;
    segments are stretched over quiet gaps, so their length alone says nothing.
    """
    trim_env, full_env = envelope(trim, sr), envelope(full, sr)
    runs = _runs(_matched_blocks(trim_env, full_env))
    if not runs:
        return [], 0.0
    n_blocks = max(1, int((len(trim_env) / FRAMES_PER_SEC - BLOCK_SEC) / BLOCK_HOP_SEC) + 1)
    coverage = min(1.0, sum(len(r) for r in runs) / n_blocks)
    block = int(BLOCK_SEC * FRAMES_PER_SEC)
    offsets = [int(round(np.median([b[1] for b in r]) * FRAMES_PER_SEC)) for r in runs]
    bounds = [0]
    for i in range(len(runs) - 1):
        lo = int(runs[i][-1][0] * FRAMES_PER_SEC)
        hi = int(runs[i + 1][0][0] * FRAMES_PER_SEC) + block
        bounds.append(_cut_between(trim_env, full_env, lo, hi, offsets[i], offsets[i + 1]))
    bounds.append(len(trim_env))
    segments = [
        Segment(
            start=bounds[i] / FRAMES_PER_SEC,
            end=bounds[i + 1] / FRAMES_PER_SEC,
            offset=offsets[i] / FRAMES_PER_SEC,
            score=float(np.mean([b[2] for b in runs[i]])),
        )
        for i in range(len(runs))
    ]
    for seg in segments:
        seg.offset = _refine_offset(trim, full, seg)
    segments[-1].end = len(trim) / sr
    return segments, coverage


def cut_track(track: np.ndarray, sr: int, segments: list[Segment], duration: float) -> np.ndarray:
    """The full clip's track laid onto the trimmed timeline; unmatched time stays silent."""
    out = np.zeros(int(round(duration * sr)), dtype=track.dtype)
    for seg in segments:
        d0, d1 = int(round(seg.start * sr)), min(len(out), int(round(seg.end * sr)))
        s0 = int(round((seg.start + seg.offset) * sr))
        s0c = max(0, s0)
        n = min(d1 - d0 - (s0c - s0), len(track) - s0c)
        if n > 0:
            out[d0 + (s0c - s0) : d0 + (s0c - s0) + n] = track[s0c : s0c + n]
    return out


def cut_events(events: list[dict], segments: list[Segment]) -> list[dict]:
    """Events whose peak survives the trim, moved onto the trimmed timeline."""
    out = []
    for seg in segments:
        lo, hi = seg.start + seg.offset, seg.end + seg.offset
        for ev in events:
            if not lo <= ev["peak_sec"] < hi:
                continue
            moved = dict(ev)
            moved["peak_sec"] = round(ev["peak_sec"] - seg.offset, 3)
            moved["start_sec"] = round(max(seg.start, ev["start_sec"] - seg.offset), 3)
            moved["end_sec"] = round(min(seg.end, ev["end_sec"] - seg.offset), 3)
            moved["source_peak_sec"] = ev["peak_sec"]
            out.append(moved)
    out.sort(key=lambda e: e["start_sec"])
    for i, ev in enumerate(out, start=1):
        ev["event_id"] = f"event_{i:03d}"
    return out


def _load_mono(path: Path, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(int(file_sr), sr)
        audio = resample_poly(audio, sr // g, int(file_sr) // g)
    return audio.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("full_result", type=Path, help="extracted ZIP of the full video's run")
    parser.add_argument("trimmed_video", type=Path, help="the trimmed video")
    parser.add_argument("out", type=Path, help="folder to write (created)")
    parser.add_argument("--zip", action="store_true", help="also write OUT.zip")
    args = parser.parse_args()

    full_dir = args.full_result
    source_wav = full_dir / "debug" / "audio_preparation" / "source_audio.wav"
    events_json = full_dir / "debug" / "context_detector" / "events.json"
    for path in (source_wav, events_json, args.trimmed_video):
        if not path.is_file():
            sys.exit(f"Missing: {path}")

    with tempfile.TemporaryDirectory() as td:
        trim_wav = extract_audio_from_video(args.trimmed_video, Path(td) / "trim.wav", sr=ALIGN_SR)
        trim = _load_mono(trim_wav, ALIGN_SR)
    full = _load_mono(source_wav, ALIGN_SR)
    duration = len(trim) / ALIGN_SR

    segments, covered = find_segments(trim, full)
    print(f"trimmed clip {duration:.1f} s, full clip {len(full) / ALIGN_SR:.1f} s, {covered:.0%} of it matched")
    for s in segments:
        print(
            f"  trimmed {s.start:7.2f}-{s.end:7.2f} s  =  full {s.start + s.offset:7.2f}-{s.end + s.offset:7.2f} s"
            f"  (match {s.score:.2f})"
        )
    if not segments or covered < MIN_COVERAGE:
        sys.exit(
            "The trimmed video's sound does not match the full clip's "
            f"({covered:.0%} matched). It must be the original soundtrack: "
            "export without added music, volume changes or a new mix."
        )

    out = args.out
    (out / "debug" / "context_detector").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.trimmed_video, out / "video.mp4")
    for wav in sorted(full_dir.glob("algorithm_*.wav")):
        data, sr = sf.read(wav, always_2d=False)
        info = sf.info(wav)
        sf.write(out / wav.name, cut_track(data, sr, segments, duration), sr, subtype=info.subtype)

    payload = json.loads(events_json.read_text(encoding="utf-8"))
    events = cut_events(payload.get("events", []), segments)
    payload["events"] = events
    payload["trimmed_from"] = {
        "full_result": full_dir.name,
        "trimmed_video": args.trimmed_video.name,
        "segments": [
            {"start_sec": round(s.start, 3), "end_sec": round(s.end, 3), "offset_sec": round(s.offset, 4)}
            for s in segments
        ],
    }
    (out / "debug" / "context_detector" / "events.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    kept = sum(1 for e in events if e.get("included_in_gate"))
    print(f"wrote {out}: {len(events)} events ({kept} felt), {len(list(out.glob('algorithm_*.wav')))} haptic WAVs")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip: {archive}")


if __name__ == "__main__":
    main()
