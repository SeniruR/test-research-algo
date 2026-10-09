"""Record or check output hashes for a deterministic pipeline run.

Uses manual events (no ML) and a pre-extracted source WAV (no ffmpeg), so the
same inputs always produce the same WAVs. Run before and after a refactor:

    python scripts/regression_baseline.py record
    python scripts/regression_baseline.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "sample"
VIDEO = SAMPLE / "war_tank_001.mp4"
SOURCE_WAV = SAMPLE / "haptic_candidates (35)" / "source_audio.wav"
EVENTS_JSON = SAMPLE / "haptic_candidates (35)" / "events.json"
BASELINE = ROOT / "tests" / "baseline" / "war_tank_001_hashes.json"

GATE_CATEGORIES = ["gunshot", "explosion", "smash", "car_crash", "vehicle"]
SCENARIOS = {
    "default": {"continuous": False},
    "continuous": {"continuous": True},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_source(_video, output_path, sr=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_WAV, output_path)
    return output_path


def _run_legacy(scenario: dict, out_dir: Path) -> dict[str, Path]:
    from haptic_gt import pipeline

    pipeline.extract_audio_from_video = _copy_source
    tracks = pipeline.generate_candidate_tracks(
        VIDEO,
        out_dir,
        from_video=True,
        enable_context_detection=False,
        manual_events=EVENTS_JSON,
        gate_categories=GATE_CATEGORIES,
        continuous_haptics=scenario["continuous"],
    )
    return tracks.save_all()


def _run(scenario: dict, out_dir: Path) -> dict[str, Path]:
    try:
        from haptic_gt.core import runner
    except ImportError:
        return _run_legacy(scenario, out_dir)
    from haptic_gt.core.config import load_pipeline_config

    runner.extract_audio_from_video = _copy_source
    config = load_pipeline_config(
        overrides={
            "detector": {
                "name": "manual_events",
                "params": {"events": str(EVENTS_JSON), "gate_categories": GATE_CATEGORIES},
            },
            "synthesis": {"continuous": {"enabled": scenario["continuous"]}},
        }
    )
    result = runner.run_pipeline(config, VIDEO, out_dir)
    return result.save_all()


def _events_digest(events_json: Path) -> str:
    payload = json.loads(events_json.read_text(encoding="utf-8"))
    keys = ("category", "start_sec", "peak_sec", "end_sec", "included_in_gate")
    rows = [{k: ev.get(k) for k in keys} for ev in payload.get("events", [])]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def collect() -> dict:
    for path in (VIDEO, SOURCE_WAV, EVENTS_JSON):
        if not path.is_file():
            raise SystemExit(f"missing reference input: {path}")
    out: dict = {
        "inputs": {p.name: _sha256(p) for p in (VIDEO, SOURCE_WAV, EVENTS_JSON)},
        "scenarios": {},
    }
    for name, scenario in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as tmp:
            saved = _run(scenario, Path(tmp))
            hashes = {
                Path(p).name: _sha256(Path(p))
                for key, p in saved.items()
                if Path(p).suffix == ".wav" and key != "source_audio"
            }
            if "events_json" in saved:
                hashes["events.json#events"] = _events_digest(Path(saved["events_json"]))
            out["scenarios"][name] = dict(sorted(hashes.items()))
            print(f"[{name}]")
            for file_name, digest in out["scenarios"][name].items():
                print(f"  {digest[:16]}  {file_name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "check"])
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Output file names allowed to differ (e.g. algorithm_c_pitch_matching.wav)",
    )
    args = parser.parse_args()

    current = collect()
    if args.mode == "record":
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print("recorded", BASELINE)
        return 0

    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    if expected["inputs"] != current["inputs"]:
        print("reference inputs changed; re-record the baseline")
        return 2
    failures = []
    for scenario, files in expected["scenarios"].items():
        got = current["scenarios"].get(scenario, {})
        for file_name, digest in files.items():
            if file_name in args.ignore:
                continue
            if got.get(file_name) != digest:
                failures.append(f"{scenario}/{file_name}")
        for file_name in sorted(set(got) - set(files)):
            failures.append(f"{scenario}/{file_name} (unexpected output)")
    if failures:
        print("MISMATCH:")
        for item in failures:
            print("  ", item)
        return 1
    print("all outputs match the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
