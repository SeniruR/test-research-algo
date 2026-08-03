"""Move events timeline cell after generation and add OUTPUT_DIR fallback."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "haptic_groundtruth_colab.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    return "".join(cell.get("source", []))


viz_idx = next(i for i, c in enumerate(cells) if "Events timeline visualization" in src(c))
gen_idx = next(
    i
    for i, c in enumerate(cells)
    if "generate_candidate_tracks" in src(c) and "%%time" in src(c)
)

viz_cell = cells.pop(viz_idx)
if viz_idx < gen_idx:
    gen_idx -= 1
cells.insert(gen_idx + 1, viz_cell)

viz_source = """# Events timeline visualization (run after Drive mount + generation cells)
import json
from pathlib import Path
import matplotlib.pyplot as plt

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/content/drive/MyDrive/haptic-groundtruth/output")

events_path = Path(OUTPUT_DIR) / "events.json"
if events_path.exists():
    data = json.loads(events_path.read_text())
    events = data.get("events", [])
    if events:
        fig, ax = plt.subplots(figsize=(12, 2 + len(events) * 0.3))
        colors = {
            "weather": "tab:blue",
            "gunshot": "tab:red",
            "vehicle": "tab:orange",
            "human_activity": "tab:green",
        }
        for i, ev in enumerate(events):
            c = colors.get(ev["category"], "tab:gray")
            ax.barh(
                i,
                ev["end_sec"] - ev["start_sec"],
                left=ev["start_sec"],
                height=0.6,
                color=c,
                alpha=0.7,
            )
            ax.plot(ev["peak_sec"], i, "k|", markersize=12)
            ax.text(
                ev["end_sec"],
                i,
                f" {ev['label']} ({ev['confidence']:.0%})",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel("Time (s)")
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels([e["category"] for e in events])
        ax.set_title("Detected events timeline")
        plt.tight_layout()
        plt.show()
    else:
        print("No events in events.json — haptics used full source audio.")
else:
    print("Run the generation cell first to create events.json")
"""

for c in cells:
    if "Events timeline visualization" in src(c):
        c["source"] = [line + "\n" for line in viz_source.split("\n")]
        if c["source"] and c["source"][-1] == "\n":
            c["source"].pop()
        break

nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Fixed {nb_path.name}: timeline cell now after generation (index {gen_idx + 1})")
