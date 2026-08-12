"""Regenerate the full Colab notebook from haptic_gt source files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "haptic_gt"
NOTEBOOK = ROOT / "notebooks" / "haptic_groundtruth_colab.ipynb"

BOOTSTRAP_HEADER = '''import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/haptic-groundtruth")
PKG_DIR = PROJECT_ROOT / "haptic_gt"
PKG_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
'''

BOOTSTRAP_FOOTER = '''
}

for name, source in FILES.items():
    target = PKG_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")

sys.path.insert(0, str(PROJECT_ROOT))
print("Installed haptic_gt at", PKG_DIR)
print("Modules:", ", ".join(sorted(FILES)))
from haptic_gt.context.encoders import AST_MODEL_ID, VIVIT_MODEL_ID
print("AST model:", AST_MODEL_ID)
print("ViViT model:", VIVIT_MODEL_ID)
'''

# ---------------------------------------------------------------------------
# All non-bootstrap cells as plain Python source strings
# ---------------------------------------------------------------------------

CELL_INSTALL = """\
# Install system + Python dependencies
!apt-get -qq install -y ffmpeg > /dev/null
!pip install -q numpy scipy librosa soundfile audioread resampy torch torchaudio matplotlib mosqito pyyaml transformers accelerate decord av opencv-python Pillow
# Optional: PANNs gives true framewise SED (~10 ms frames). Without it the
# detector falls back to densely-strided AST (100 ms frames). GPU runtime advised.
!pip install -q panns-inference
"""

CELL_WORKSPACE = """\
# Workspace folders on this Colab VM (no Google Drive needed)
from pathlib import Path

WORK_DIR = Path("/content/haptic-workspace")
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Upload a video in the next cell.")
print("Results will be written to:", OUTPUT_DIR)
print("Download the ZIP at the end — Colab deletes /content when the session ends.")
"""

CELL_UPLOAD = """\
from google.colab import files
from pathlib import Path
from IPython.display import Audio, display
from haptic_gt.pipeline import generate_candidate_tracks

print("Choose a video file to upload...")
uploaded = files.upload()
video_name = next(iter(uploaded))
video_path = Path("/content") / video_name

print("Using video:", video_path)
print("Output folder:", OUTPUT_DIR)
"""

CELL_GENERATE = """\
%%time
import json
import soundfile as sf
from pathlib import Path
from haptic_gt.pipeline import OUTPUT_NAMES, generate_candidate_tracks

CONTENT_TYPE = "game"
ENABLE_CONTEXT = True  # set False to skip AST/ViViT and re-run Sound2Hap only
# Categories to include in gated haptics
GATE_CATEGORIES = ["weather", "gunshot", "explosion", "vehicle", "human_activity"]

# Auto-detect timing (model alone). Only set a dict for HITL override.
MANUAL_EVENTS = None
# Example override (optional):
# MANUAL_EVENTS = {
#     "category": "explosion",
#     "start_sec": 0.22,
#     "peak_sec": 0.24,
#     "end_sec": 3.66,
# }

# Replace auto vehicle with your rumble marks (keeps auto gunshot/explosion).
# Calib: true rumbles often rise ~1.11–1.23 while bed FPs rise ~1.38+ — not separable by RMS.
MANUAL_RUMBLE_PEAKS = None
# MANUAL_RUMBLE_PEAKS = [6.0, 24.0, 29.0, 30.0, 31.0, 31.9, 41.0, 42.0, 43.0, 45.0, 48.0, 49.0, 51.0]

source_wav = OUTPUT_DIR / OUTPUT_NAMES["source_audio"]
if source_wav.exists() and not ENABLE_CONTEXT and MANUAL_EVENTS is None:
  print("Reusing existing source audio:", source_wav)
  input_path = source_wav
  from_video = False
else:
  input_path = video_path
  from_video = True

tracks = generate_candidate_tracks(
    input_path,
    OUTPUT_DIR,
    from_video=from_video,
    content_type=CONTENT_TYPE,
    enable_context_detection=ENABLE_CONTEXT,
    gate_categories=GATE_CATEGORIES,
    manual_events=MANUAL_EVENTS,
    manual_rumble_peaks=MANUAL_RUMBLE_PEAKS,
)
saved = tracks.save_all()

src_audio, src_sr = sf.read(saved["source_audio"])
print(f"Source: {len(src_audio)/src_sr:.1f}s @ {src_sr} Hz")
print(f"Haptic input: {tracks.haptic_input_wav}")
print(f"No events detected: {tracks.no_events_detected}")
print(f"No haptic events (gate): {tracks.no_haptic_events}")
print(f"Gate categories: {tracks.gate_categories_used}")
if tracks.events_json and tracks.events_json.exists():
    print(f"Events: {tracks.events_json}")
    print(json.dumps(json.loads(tracks.events_json.read_text()), indent=2)[:2000])

print("Saved files:")
missing = []
for name, path in saved.items():
    ok = path.exists()
    print(f"  {name}: {path} [{'OK' if ok else 'MISSING'}]")
    if not ok:
        missing.append(name)

if tracks.no_haptic_events:
    print("No gate-eligible events — haptic WAVs skipped (see events.json).")
elif missing:
    raise RuntimeError(
        "Generation incomplete — missing outputs: "
        + ", ".join(missing)
        + ". Scroll up for the first error, fix it, then re-run this cell."
    )
print("Generation complete.")
"""

CELL_EVENTS_VIZ = """\
# Events timeline visualization (run after generation cell)
import json
from pathlib import Path
import matplotlib.pyplot as plt

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

events_path = Path(OUTPUT_DIR) / "events.json"
if events_path.exists():
    data = json.loads(events_path.read_text())
    events = data.get("events", [])
    if events:
        fig, ax = plt.subplots(figsize=(12, 2 + len(events) * 0.3))
        colors = {
            "weather": "tab:blue",
            "gunshot": "tab:red",
            "explosion": "tab:purple",
            "vehicle": "tab:orange",
            "human_activity": "tab:green",
        }
        for i, ev in enumerate(events):
            c = colors.get(ev["category"], "tab:gray")
            ax.barh(i, ev["end_sec"] - ev["start_sec"], left=ev["start_sec"],
                    height=0.6, color=c, alpha=0.7)
            ax.plot(ev["peak_sec"], i, "k|", markersize=12)
            ax.text(ev["end_sec"], i,
                    f" {ev['label']} ({ev['confidence']:.0%})",
                    va="center", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels([e["category"] for e in events])
        ax.set_title("Detected events timeline")
        plt.tight_layout()
        plt.show()
    else:
        print("No events in events.json.")
else:
    print("Run the generation cell first to create events.json")
"""

CELL_RUMBLE_CALIB = """\
# Calibrate vehicle rumble against YOUR marked peaks (span coverage + local RMS).
# Scoring only: fill TRUTH_RUMBLE_TIMES, run after generation. Auto gates on loudness.
from pathlib import Path
import json
import soundfile as sf
from haptic_gt.context.rumble_calib import calibrate_rumble_thresholds, format_calibration_table
from haptic_gt.pipeline import OUTPUT_NAMES

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

# Rumble times you hear in THIS clip (seconds). Empty by default on purpose:
# marks left over from another clip score this one against the wrong video and
# every number in the report comes out zero.
# GRADING KEY ONLY -- this does not steer detection. The detector already ran,
# unaided, in the generate cell; these marks just say where you expected it to
# fire so the report can count hits and misses. (The HITL override that *does*
# change detection is MANUAL_RUMBLE_PEAKS in the generate cell, left at None.)
TRUTH_RUMBLE_TIMES = []

source = OUTPUT_DIR / OUTPUT_NAMES["source_audio"]
events_path = OUTPUT_DIR / OUTPUT_NAMES["events_json"]
if not source.exists() or not events_path.exists():
    raise FileNotFoundError("Run the generation cell first (need source_audio.wav + events.json).")

report = calibrate_rumble_thresholds(
    source,
    TRUTH_RUMBLE_TIMES,
    events_path,
    match_tolerance_sec=1.0,
)
print(format_calibration_table(report))
print(
    "\\nAuto rumble windows are loud RMS islands (not AST start/end slabs). "
    "A 29–32s island hits every mark inside it; quiet gaps should stay unmarked. "
    "These marks grade the run; they never feed the detector."
)
(OUTPUT_DIR / "rumble_calibration.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("Wrote", OUTPUT_DIR / "rumble_calibration.json")

# Dense 100 ms scan. The whole clip by default: a fixed window belongs to
# whichever video it was typed for, and on the next clip it scans a second of
# nothing. Narrow it only to zoom in on a stretch you are arguing about.
from haptic_gt.context.rumble_calib import scan_rumble_timeline, format_timeline_scan
import matplotlib.pyplot as plt

_dur = float(sf.info(str(source)).duration)
SCAN_START = 0.0
SCAN_END = _dur
SCAN_HOP = 0.1  # 100 ms. Set 0.01 for 10 ms (more rows).

SCAN_START = max(0.0, min(SCAN_START, _dur))
SCAN_END = min(SCAN_END, _dur)

scan = scan_rumble_timeline(
    source,
    start_sec=SCAN_START,
    end_sec=SCAN_END,
    hop_sec=SCAN_HOP,
    events_json=events_path,
    manual_peaks_sec=TRUTH_RUMBLE_TIMES,
)
print(format_timeline_scan(scan, only_manual_windows=bool(TRUTH_RUMBLE_TIMES)))
print("\\n(Full scan saved; the table shows ticks near your marks when you set any.)")
SCAN_NAME = "rumble_scan.json"
(OUTPUT_DIR / SCAN_NAME).write_text(json.dumps(scan, indent=2), encoding="utf-8")

ts = [s["t_sec"] for s in scan["samples"]]
rises = [s["rms_rise_ratio"] or 0.0 for s in scan["samples"]]
rmss = [s["local_rms"] for s in scan["samples"]]
thr = scan["summary"].get("salience_threshold")
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(ts, rises, lw=1)
ax[0].axhline(1.10, color="gray", ls="--", lw=0.8, label="1.10")
ax[0].axhline(1.25, color="orange", ls="--", lw=0.8, label="1.25")
ax[0].axhline(1.35, color="red", ls="--", lw=0.8, label="1.35")
for m in TRUTH_RUMBLE_TIMES:
    if SCAN_START <= m <= SCAN_END:
        ax[0].axvline(m, color="green", alpha=0.4, lw=1)
ax[0].set_ylabel("RMS rise (unused)")
ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title(f"Rumble scan {SCAN_START:.1f}–{SCAN_END:.1f}s @ {SCAN_HOP}s")
ax[1].plot(ts, rmss, lw=1, color="tab:blue")
if thr is not None:
    ax[1].axhline(thr, color="purple", ls="--", lw=1, label=f"salience {thr:.3f}")
for m in TRUTH_RUMBLE_TIMES:
    if SCAN_START <= m <= SCAN_END:
        ax[1].axvline(m, color="green", alpha=0.4, lw=1)
ax[1].set_ylabel("local RMS (auto gate)")
ax[1].set_xlabel("Time (s)")
ax[1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
print("Wrote", OUTPUT_DIR / SCAN_NAME)

# What the loudness gate decided, so a missing rumble is diagnosable: one floor
# per scene, plus every span it dropped and how far under the floor it was.
gate = json.loads(events_path.read_text(encoding="utf-8")).get("sustained_gate") or {}
if gate:
    print("\\nrumble gate")
    print("  clip floor {0:.4f}  proposals {1}  kept {2}".format(
        gate.get("clip_threshold", 0.0), gate.get("proposals", 0), gate.get("kept", 0)
    ))
    for sc in gate.get("scenes", []):
        print("  scene {0:6.2f}-{1:6.2f}s  floor {2:.4f}".format(
            sc["start_sec"], sc["end_sec"], sc["threshold"]
        ))
    for d in gate.get("dropped", []):
        print("  dropped {0:.2f}-{1:.2f}s  local RMS {2:.4f} vs floor {3:.4f}  ({4})".format(
            d["start_sec"], d["end_sec"], d["local_rms"], d["scene_threshold"], d["reason"]
        ))
    if not gate.get("dropped"):
        print("  nothing dropped: rumble you cannot feel was never proposed, not gated out")
"""

CELL_SHOT_CALIB = """\
# Calibrate cannon / gunshot timing against YOUR marked shot times.
# Shows, for each mark, the nearest real attack in the audio and how strong it is
# relative to this clip's confident blasts. Use it to tell a detector error from
# a mis-typed mark.
from pathlib import Path
import json
import matplotlib.pyplot as plt
from haptic_gt.context.shot_calib import (
    calibrate_shot_times,
    format_shot_calibration_table,
    format_shot_scan,
    scan_shot_attacks,
)
from haptic_gt.pipeline import OUTPUT_NAMES

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

# Shot times you hear in THIS clip (seconds). Empty by default: another clip's
# times score this one against the wrong video, and the whole report reads as a
# detector failure when the marks are simply not from this edit.
# GRADING KEY ONLY: detection already happened, unaided, in the generate cell.
TRUTH_SHOT_TIMES = []

source = OUTPUT_DIR / OUTPUT_NAMES["source_audio"]
events_path = OUTPUT_DIR / OUTPUT_NAMES["events_json"]
if not source.exists() or not events_path.exists():
    raise FileNotFoundError("Run the generation cell first (need source_audio.wav + events.json).")

report = calibrate_shot_times(
    source,
    TRUTH_SHOT_TIMES,
    events_path,
    match_tolerance_sec=0.35,
)
print(format_shot_calibration_table(report))
(OUTPUT_DIR / "shot_calibration.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

# Every strong attack in the clip, whether or not it became an event.
scan = scan_shot_attacks(source, top_n=40)
print()
print(format_shot_scan(scan))
(OUTPUT_DIR / "shot_scan.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")

ts = [a["t_sec"] for a in scan["attacks"]]
vals = [a["flux_rel_max"] for a in scan["attacks"]]
fig, ax = plt.subplots(figsize=(12, 3))
ax.stem(ts, vals, basefmt=" ")
for m in TRUTH_SHOT_TIMES:
    ax.axvline(m, color="green", alpha=0.45, lw=1)
for e in json.loads(events_path.read_text(encoding="utf-8"))["events"]:
    if e["category"] in ("explosion", "gunshot"):
        ax.axvline(e["peak_sec"], color="red", ls="--", alpha=0.6, lw=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("attack / clip max")
ax.set_title("Flux attacks — green = your marks, red dashed = detected shots")
plt.tight_layout()
plt.show()
print("Wrote", OUTPUT_DIR / "shot_calibration.json", "and", OUTPUT_DIR / "shot_scan.json")
"""

CELL_SED_EVAL = """\
# DCASE-style scoring of events.json against your ground truth.
# Event-based F1 uses a one-to-one onset match inside a collar (DCASE Task 4 uses
# 200 ms); segment-based F1 ignores onset jitter and asks "did we find it at all".
from pathlib import Path
import json
from haptic_gt.context.taxonomy import load_taxonomy
from haptic_gt.eval.sed_metrics import evaluate_events, format_evaluation
from haptic_gt.pipeline import OUTPUT_NAMES
import soundfile as sf

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

# Ground truth comes from the marks you entered in the calibration cells above,
# so one clip cannot be scored against another clip's marks. Run those first, or
# set GROUND_TRUTH here by hand.
GROUND_TRUTH = {
    "explosion": list(globals().get("TRUTH_SHOT_TIMES") or []),
    "vehicle": list(globals().get("TRUTH_RUMBLE_TIMES") or []),
}
GROUND_TRUTH = {k: v for k, v in GROUND_TRUTH.items() if v}

source = OUTPUT_DIR / OUTPUT_NAMES["source_audio"]
events_path = OUTPUT_DIR / OUTPUT_NAMES["events_json"]
if not source.exists() or not events_path.exists():
    raise FileNotFoundError("Run the generation cell first.")

info = sf.info(str(source))
duration = float(info.duration)
events = json.loads(events_path.read_text(encoding="utf-8"))
print("detector:", events.get("detector", {}))
print("clip duration: {0:.2f}s".format(duration))
print("ground truth:", {k: len(v) for k, v in GROUND_TRUTH.items()})

detected_cats = {e["category"] for e in events.get("events", [])}
for cat, marks in GROUND_TRUTH.items():
    if any(m > duration for m in marks):
        print(
            "WARNING: {0} marks fall past the end of this clip -- these look like "
            "another video's marks, scores will be meaningless.".format(cat)
        )
    if cat not in detected_cats:
        print(
            "WARNING: {0} marks exist but nothing of that category was detected. "
            "If this clip has no {0}, clear those marks -- otherwise every one "
            "counts as a miss and drags F1 to 0.".format(cat)
        )

if not GROUND_TRUTH:
    print(
        "Nothing to score yet. Put what YOU hear in this clip into "
        "TRUTH_SHOT_TIMES / TRUTH_RUMBLE_TIMES in the calibration cells above and "
        "re-run them, then run this cell. Detection does not need them; scoring "
        "does. Detected so far:"
    )
    for e in events.get("events", []):
        print("  {0:14} {1:7.2f}-{2:7.2f}s  peak {3:7.2f}s".format(
            e["category"], e["start_sec"], e["end_sec"], e["peak_sec"]
        ))
else:
    # Rumble marks are moments inside a burst, not onsets, so they get coverage
    # (recall + how much of the clip vibrates) instead of an onset-collar F1.
    sustained = tuple(
        name for name, cfg in load_taxonomy().categories.items() if not cfg.impulsive
    )
    report = evaluate_events(
        GROUND_TRUTH,
        events,
        duration_sec=duration,
        collars_sec=(0.2, 0.5),
        segment_sec=1.0,
        sustained_categories=sustained,
    )
    print()
    print(format_evaluation(report))
    (OUTPUT_DIR / "sed_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print()
    print("0.2 s is the DCASE onset collar. Hand marks drift ~0.5 s, so read both.")
    print("Wrote", OUTPUT_DIR / "sed_evaluation.json")
"""

CELL_PLAYBACK = """\
from pathlib import Path
from IPython.display import Audio, display
from haptic_gt.pipeline import OUTPUT_NAMES

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

labels = {
    "source_audio": "Source audio",
    "algorithm_a_perception_mapping": "A — Perception mapping",
    "algorithm_b_frequency_shifting": "B — Frequency shifting",
    "algorithm_c_pitch_matching": "C — Pitch matching",
    "algorithm_d_haptic_gen": "D — HapticGen",
}

paths = {}
if "saved" in globals():
    paths.update({k: v for k, v in saved.items() if k in labels and v.exists()})
for key in labels:
    if key not in paths:
        candidate = OUTPUT_DIR / OUTPUT_NAMES[key]
        if candidate.exists():
            paths[key] = candidate

missing = [labels[k] for k in labels if k not in paths]
if missing:
    print("Some tracks are missing — re-run the generation cell:")
    for name in missing:
        print(" -", name)
    on_disk = sorted(p.name for p in OUTPUT_DIR.glob("*"))
    print(f"\\nCurrently in {OUTPUT_DIR}:", on_disk or "(empty)")
    if not paths:
        raise RuntimeError("No audio outputs found yet. Run the generation cell first.")

for key, label in labels.items():
    if key not in paths:
        continue
    print(label)
    display(Audio(str(paths[key])))
"""

CELL_DOWNLOAD = """\
import shutil
from pathlib import Path
from google.colab import files

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/haptic-workspace/output")

zip_base = OUTPUT_DIR.parent / "haptic_candidates"
zip_path = Path(shutil.make_archive(str(zip_base), "zip", OUTPUT_DIR))
print("ZIP created:", zip_path)

files.download(str(zip_path))
print("Download started.")
"""

MARKDOWN_INTRO = """\
# Haptic Ground Truth — Colab Pipeline

Same four algorithms as **[Sound2Hap](https://github.com/Iris1215/Sound2Hap)** (CHI 2026), with **frozen multimodal context detection** before haptic synthesis.

Convert **3–5 minute** video into **gated candidate haptic tracks** for human-in-the-loop evaluation.

**Runtime:** **T4 GPU** recommended for context detection (AST + ViViT). Sound2Hap A–D can run on CPU.

**Output:** mono **8 kHz** haptic WAV files + `events.json`

**Setup:** Run all cells top-to-bottom. Upload a video when prompted — no Google Drive needed.

| Phase | Component |
|-------|-----------|
| **1** | Tokenization (100Hz) + Frozen Context Detectors + AST/ViViT encoders |
| **2** | Frozen fusion → `events.json` → gated audio |
| **3** | Sound2Hap A–D |\
"""

MARKDOWN_HITL = """\
## Human-in-the-loop (next step)

1. Play **source audio** (44.1 kHz) in headphones while feeling each **8 kHz** candidate on haptic hardware.
2. Rate **realism** and **similarity** (e.g. 1–7 Likert) per algorithm — same protocol as [Sound2Hap](https://sound2hap.netlify.app/).
3. If agreement < threshold, tune parameters in `haptic_gt/algorithms/*.py` and re-run.
4. Approved tracks become your **ground truth dataset**.\
"""

# ---------------------------------------------------------------------------


def escape_for_triple_quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def collect_modules() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        files.append((p.relative_to(PKG).as_posix(), p))
    for p in sorted(PKG.rglob("*.yaml")):
        files.append((p.relative_to(PKG).as_posix(), p))
    return files


def make_code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build_bootstrap_source(modules: list[tuple[str, Path]]) -> str:
    lines = [BOOTSTRAP_HEADER]
    for i, (rel, path) in enumerate(modules):
        content = path.read_text(encoding="utf-8")
        escaped = escape_for_triple_quote(content)
        comma = "," if i < len(modules) - 1 else ""
        lines.append(f'    "{rel}": """{escaped}"""{comma}\n')
    lines.append(BOOTSTRAP_FOOTER)
    return "".join(lines)


def main() -> None:
    modules = collect_modules()
    bootstrap_source = build_bootstrap_source(modules)

    cells = [
        make_markdown_cell(MARKDOWN_INTRO, "a8f0a7bd"),
        make_code_cell(CELL_INSTALL, "e4d3bed2"),
        make_code_cell(bootstrap_source, "8fd56542"),
        make_code_cell(CELL_WORKSPACE, "48cf935d"),
        make_code_cell(CELL_UPLOAD, "9c15aaa0"),
        make_code_cell(CELL_GENERATE, "f6c03115"),
        make_code_cell(CELL_EVENTS_VIZ, "45404a15"),
        make_code_cell(CELL_RUMBLE_CALIB, "b7e2c901"),
        make_code_cell(CELL_SHOT_CALIB, "c31d8a45"),
        make_code_cell(CELL_SED_EVAL, "d47b91e0"),
        make_code_cell(CELL_PLAYBACK, "e8083503"),
        make_code_cell(CELL_DOWNLOAD, "f9703b79"),
        make_markdown_cell(MARKDOWN_HITL, "d0fce921"),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK} ({len(modules)} modules)")

    # Validate
    try:
        json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        print("JSON validation: OK")
    except Exception as e:
        print(f"JSON validation FAILED: {e}")


if __name__ == "__main__":
    main()
