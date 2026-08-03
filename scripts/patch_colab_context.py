"""Patch Colab notebook for context detection."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "haptic_groundtruth_colab.ipynb"


def set_cell_source(cells: list, predicate, new_source: str) -> bool:
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if predicate(src):
            cell["source"] = [
                line + "\n" for line in new_source.splitlines()
            ]
            if not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            return True
    return False


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    set_cell_source(
        cells,
        lambda s: "apt-get" in s and "pip install" in s,
        """# Install system + Python dependencies
!apt-get -qq install -y ffmpeg > /dev/null
!pip install -q numpy scipy librosa soundfile audioread resampy torch torchaudio matplotlib mosqito pyyaml transformers accelerate decord av opencv-python Pillow""",
    )

    # Insert detection viz cell before generation if not present
    has_detect_viz = any(
        "events timeline" in "".join(c.get("source", [])).lower()
        for c in cells
    )

    generation_src = """%%time
import json
import soundfile as sf

CONTENT_TYPE = "game"
ENABLE_CONTEXT = True

tracks = generate_candidate_tracks(
    video_path,
    OUTPUT_DIR,
    from_video=True,
    content_type=CONTENT_TYPE,
    enable_context_detection=ENABLE_CONTEXT,
)
saved = tracks.save_all()

src_audio, src_sr = sf.read(saved["source_audio"])
print(f"Source: {len(src_audio)/src_sr:.1f}s @ {src_sr} Hz")
print(f"Haptic input: {tracks.haptic_input_wav}")
print(f"No events detected: {tracks.no_events_detected}")
if tracks.events_json and tracks.events_json.exists():
    print(f"Events: {tracks.events_json}")
    print(json.dumps(json.loads(tracks.events_json.read_text()), indent=2)[:2000])
print("Saved files:")
for name, path in saved.items():
    print(f"  {name}: {path}")"""

    set_cell_source(
        cells,
        lambda s: "generate_candidate_tracks" in s and "%%time" in s,
        generation_src,
    )

    if not has_detect_viz:
        viz_cell = {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Events timeline visualization\n",
                "import json\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "events_path = OUTPUT_DIR / 'events.json'\n",
                "if events_path.exists():\n",
                "    data = json.loads(events_path.read_text())\n",
                "    events = data.get('events', [])\n",
                "    if events:\n",
                "        fig, ax = plt.subplots(figsize=(12, 2 + len(events) * 0.3))\n",
                "        colors = {'weather': 'tab:blue', 'gunshot': 'tab:red', 'vehicle': 'tab:orange', 'human_activity': 'tab:green'}\n",
                "        for i, ev in enumerate(events):\n",
                "            c = colors.get(ev['category'], 'tab:gray')\n",
                "            ax.barh(i, ev['end_sec'] - ev['start_sec'], left=ev['start_sec'], height=0.6, color=c, alpha=0.7)\n",
                "            ax.plot(ev['peak_sec'], i, 'k|', markersize=12)\n",
                "            ax.text(ev['end_sec'], i, f\" {ev['label']} ({ev['confidence']:.0%})\", va='center', fontsize=9)\n",
                "        ax.set_xlabel('Time (s)')\n",
                "        ax.set_yticks(range(len(events)))\n",
                "        ax.set_yticklabels([e['category'] for e in events])\n",
                "        ax.set_title('Detected events timeline')\n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "    else:\n",
                "        print('No events in events.json — haptics used full source audio.')\n",
                "else:\n",
                "    print('Run generation cell first to create events.json')\n",
            ],
            "execution_count": None,
            "outputs": [],
        }
        # Insert after generation cell
        for idx, cell in enumerate(cells):
            if cell["cell_type"] == "code" and "generate_candidate_tracks" in "".join(cell.get("source", [])):
                cells.insert(idx + 1, viz_cell)
                break

    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched", NOTEBOOK)


if __name__ == "__main__":
    main()
