"""Run the tank clip locally and copy outputs into the Android demo pack."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haptic_gt.core.config import load_pipeline_config
from haptic_gt.core.runner import run_pipeline

VIDEO = ROOT / "sample" / "war_tank_001.mp4"
OUT = ROOT / "output" / "war_tank_001"
ANDROID_ASSETS = ROOT / "vibrator-android" / "app" / "src" / "main" / "assets" / "demo"
COLAB_EVENTS = Path(r"c:\Users\senir\Downloads\haptic_candidates (29)\events.json")
GATE_CATEGORIES = ["gunshot", "explosion", "smash", "car_crash", "vehicle"]


def _panns_ready() -> bool:
    ckpt = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"
    return ckpt.exists() and ckpt.stat().st_size >= 300_000_000


def main() -> int:
    if not VIDEO.exists():
        print("missing video:", VIDEO)
        return 1

    if _panns_ready():
        detector = {"name": "fusion_detector", "params": {"gate_categories": GATE_CATEGORIES}}
        print("PANNs checkpoint ready — detecting on this PC")
    elif COLAB_EVENTS.exists():
        detector = {
            "name": "manual_events",
            "params": {"events": str(COLAB_EVENTS), "gate_categories": GATE_CATEGORIES},
        }
        print("PANNs checkpoint not downloaded yet — stitching A–D from last Colab events.json")
    else:
        print("need PANNs checkpoint (~300 MB) or a Colab events.json")
        return 1

    print("running pipeline on", VIDEO)
    config = load_pipeline_config(overrides={"detector": detector})
    result = run_pipeline(config, VIDEO, OUT)
    print("events json:", result.events_json)
    print("no_haptic_events:", result.no_haptic_events)
    saved = result.save_all()
    for name, path in saved.items():
        print(" ", name, path)

    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    copies = {"video.mp4": VIDEO, **{path.name: path for path in result.outputs.values()}}
    if result.events_json is not None:
        copies["events.json"] = result.events_json
    for dest_name, src in copies.items():
        if not src.exists():
            print("skip missing", src)
            continue
        dest = ANDROID_ASSETS / dest_name
        shutil.copy2(src, dest)
        print("copied", dest, dest.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
