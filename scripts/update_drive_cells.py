import json
from pathlib import Path

nb_path = Path(r"c:\Users\senir\Projects\haptic-groundtruth\notebooks\haptic_groundtruth_colab.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cell4 = """from google.colab import files
from IPython.display import Audio, display
from haptic_gt.pipeline import generate_candidate_tracks

# Prefer a video already on THIS runner's Drive; otherwise upload into their Drive folder
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
drive_videos = sorted(
    p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS
) if INPUT_DIR.exists() else []

if drive_videos:
    print(f"Found {len(drive_videos)} video(s) in your Drive folder:")
    for i, p in enumerate(drive_videos):
        print(f"  [{i}] {p.name}")
    choice = input("Enter index to use, or press Enter to upload a new file: ").strip()
    if choice != "":
        video_path = drive_videos[int(choice)]
    else:
        print("Upload a video (saved to YOUR Drive inputs folder)...")
        uploaded = files.upload()
        video_name = next(iter(uploaded))
        src = Path("/content") / video_name
        video_path = INPUT_DIR / video_name
        video_path.write_bytes(src.read_bytes())
else:
    print("No videos in your Drive yet. Upload one (saved to YOUR Drive)...")
    uploaded = files.upload()
    video_name = next(iter(uploaded))
    src = Path("/content") / video_name
    video_path = INPUT_DIR / video_name
    video_path.write_bytes(src.read_bytes())

print("Using video:", video_path)
print("Account:", RUNNER_EMAIL)
print("Output folder (your Drive):", OUTPUT_DIR)
"""

cell7 = """import shutil

# Save ZIP on THIS runner's Drive, and also offer browser download
zip_base = DRIVE_ROOT / "haptic_candidates"
zip_path = Path(shutil.make_archive(str(zip_base), "zip", OUTPUT_DIR))
print("Saved to your Drive:", zip_path)
print("Account:", RUNNER_EMAIL)

# Optional local browser download
files.download(str(zip_path))
print("Download started.")
"""


def to_lines(s: str):
    if not s.endswith("\n"):
        s += "\n"
    return [line + "\n" for line in s.splitlines()]


nb["cells"][4]["source"] = to_lines(cell4)
nb["cells"][7]["source"] = to_lines(cell7)
nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Updated cells 4 and 7")
