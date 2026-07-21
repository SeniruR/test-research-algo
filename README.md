# Haptic Ground Truth

Generate **candidate haptic tracks** from short video audio (3–5 min, 30/60 fps) using four classical DSP algorithms, then export WAV files for human-in-the-loop evaluation.

Designed for **Google Colab (CPU)** — no GPU required.

## Pipeline

```
Video → Extract Audio (44.1 kHz mono)
     → Algorithm A (Perception Mapping)
     → Algorithm B (Frequency Shifting)
     → Algorithm C (Pitch Matching)
     → Algorithm D (HapticGen)
     → RMS-normalized candidate WAVs → Human rating → Ground truth dataset
```

## Quick start (Colab)

1. Open [`notebooks/haptic_groundtruth_colab.ipynb`](notebooks/haptic_groundtruth_colab.ipynb) in Google Colab.
2. Run all cells.
3. Upload a video when prompted (or point to a file on Google Drive).
4. Download the five WAV files from `/content/output/`:
   - `source_audio.wav`
   - `algorithm_a_perception_mapping.wav`
   - `algorithm_b_frequency_shifting.wav`
   - `algorithm_c_pitch_matching.wav`
   - `algorithm_d_haptic_gen.wav`

## Local run (optional)

Your Dell Latitude (i7-11th, 16 GB RAM) can run the same pipeline locally for 3–5 min clips.

```bash
pip install -r requirements.txt
# ffmpeg must be on PATH

python -c "
from haptic_gt.pipeline import generate_candidate_tracks
tracks = generate_candidate_tracks('sample.mp4', 'output/', from_video=True)
paths = tracks.save_all()
print(paths)
"
```

## Human-in-the-loop

After generation, evaluators listen to the source audio while feeling each candidate track on haptic hardware (or headphones as a proxy during prototyping). Rate **realism** and **similarity**; if agreement is below threshold, tune algorithm parameters in `haptic_gt/` and re-run.

## Project layout

```
haptic-groundtruth/
├── haptic_gt/
│   ├── algorithm_a.py   # Perception mapping
│   ├── algorithm_b.py   # Frequency shifting
│   ├── algorithm_c.py   # Pitch matching
│   ├── algorithm_d.py   # HapticGen
│   ├── audio_io.py      # ffmpeg extract + WAV I/O
│   ├── normalize.py     # RMS normalization
│   └── pipeline.py      # Orchestrator
├── notebooks/
│   └── haptic_groundtruth_colab.ipynb
└── requirements.txt
```

## Notes

- **GPU / TPU:** Not needed for Algorithms A–D. Use Colab CPU runtime.
- **Video length:** Tested design target is 3–5 minutes at 44.1 kHz (~8–13M samples).
- **Frame rate:** 30 or 60 fps only affects ffmpeg demux; haptics are derived from audio.
