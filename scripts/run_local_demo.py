"""Run the tank clip locally and copy outputs into the Android demo pack."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haptic_gt.pipeline import generate_candidate_tracks

VIDEO = ROOT / "sample" / "war_tank_001.mp4"
OUT = ROOT / "output" / "war_tank_001"
ANDROID_ASSETS = ROOT / "vibrator-android" / "app" / "src" / "main" / "assets" / "demo"
COLAB_EVENTS = Path(r"c:\Users\senir\Downloads\haptic_candidates (29)\events.json")


def _panns_ready() -> bool:
    ckpt = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"
    return ckpt.exists() and ckpt.stat().st_size >= 300_000_000


def main() -> int:
    if not VIDEO.exists():
        print("missing video:", VIDEO)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print("running pipeline on", VIDEO)
    kwargs: dict = dict(
        from_video=True,
        content_type="game",
        gate_categories=["weather", "gunshot", "explosion", "vehicle"],
    )
    if _panns_ready():
        kwargs["enable_context_detection"] = True
        print("PANNs checkpoint ready — detecting on this PC")
    elif COLAB_EVENTS.exists():
        kwargs["enable_context_detection"] = False
        kwargs["manual_events"] = COLAB_EVENTS
        print("PANNs checkpoint not downloaded yet — stitching A–D from last Colab events.json")
    else:
        print("need PANNs checkpoint (~300 MB) or a Colab events.json")
        return 1

    tracks = generate_candidate_tracks(VIDEO, OUT, **kwargs)
    print("events json:", tracks.events_json)
    print("no_haptic_events:", tracks.no_haptic_events)
    saved = tracks.save_all()
    for name, path in saved.items():
        print(" ", name, path)

    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    copies = {
        "video.mp4": VIDEO,
        "events.json": OUT / "events.json",
        "algorithm_a_perception_mapping.wav": OUT / "algorithm_a_perception_mapping.wav",
        "algorithm_b_frequency_shifting.wav": OUT / "algorithm_b_frequency_shifting.wav",
        "algorithm_c_pitch_matching.wav": OUT / "algorithm_c_pitch_matching.wav",
        "algorithm_d_haptic_gen.wav": OUT / "algorithm_d_haptic_gen.wav",
        "algorithm_e_rule_based.wav": OUT / "algorithm_e_rule_based.wav",
        "algorithm_e_rule_based.json": OUT / "algorithm_e_rule_based.json",
    }
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
