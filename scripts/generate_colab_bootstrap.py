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

source_wav = OUTPUT_DIR / OUTPUT_NAMES["source_audio"]
if source_wav.exists() and not ENABLE_CONTEXT:
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
