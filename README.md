# Haptic Ground Truth

Generate **candidate haptic tracks** from short video audio (3–5 min) using [Sound2Hap](https://github.com/Iris1215/Sound2Hap) algorithms, with optional **frozen multimodal context detection** before haptic synthesis.

## Pipeline

```
Video + Audio (44.1 kHz)
  → Tokenization (100Hz timeline, 16kHz audio)
  → Onset proposals (RMS transients)
  → Targeted AST + ViViT on proposal windows only
  → Event aggregation + frozen fusion
  → events.json + gated_audio.wav (selected categories)
  → Sound2Hap A–D (stitched full-timeline) → 4 × 8 kHz haptic WAVs
```

One video produces **four full-length haptic WAVs** (one per algorithm). Haptics appear only at times where detected events match your selected categories; the rest of the timeline is silent.

If **no events** match `gate_categories`, haptic generation is **skipped** (`no_haptic_events: true` in `events.json`).

## Context categories (v1)

| Category | Examples |
|----------|----------|
| `weather` | thunder, rain, wind |
| `gunshot` | gunfire |
| `explosion` | blast, artillery, fireworks |
| `vehicle` | engine, truck rumble |
| `human_activity` | chainsaw, chopping wood |

Configured in [`haptic_gt/context/taxonomy.yaml`](haptic_gt/context/taxonomy.yaml).

## Haptic category selection

Choose which detected categories contribute to haptics at run time:

```python
from haptic_gt.pipeline import generate_candidate_tracks

tracks = generate_candidate_tracks(
    "gameplay.mp4",
    "output/",
    gate_categories=["gunshot", "explosion", "vehicle", "human_activity"],
)
```

Omit `gate_categories` to use taxonomy defaults (`include_in_haptic_gate: true`, currently `gunshot` and `explosion`).

## Manual events (HITL ground truth)

Skip AST/ViViT and trust your start/peak/end marks:

```python
tracks = generate_candidate_tracks(
    "tank_cannon.mp4",
    "output/",
    manual_events={
        "category": "explosion",
        "start_sec": 0.22,
        "peak_sec": 0.24,
        "end_sec": 3.66,
    },
)
```

`manual_events` also accepts a list of events or a path to an `events.json` file.

## Models (all frozen)

| Component | Model |
|-----------|--------|
| Audio encoder / context | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Video encoder / context | `google/vivit-b-16x2-kinetics400` |

**Colab:** use **T4 GPU** for context detection; Sound2Hap A–D can run on CPU.

## Sound2Hap outputs

| File | Algorithm |
|------|-----------|
| `algorithm_a_perception_mapping.wav` | Perceptual Mapping |
| `algorithm_b_frequency_shifting.wav` | Frequency Shifting |
| `algorithm_c_pitch_matching.wav` | Pitch Matching |
| `algorithm_d_haptic_gen.wav` | HapticGen |

- Input: gated mono **44.1 kHz** (event segments only)
- Output: mono **8 kHz** haptic WAV (full video duration)

## events.json

```json
{
  "no_events_detected": false,
  "no_haptic_events": false,
  "gate_categories_used": ["gunshot", "vehicle"],
  "haptic_outputs": {
    "gated_audio": "gated_audio.wav",
    "algorithm_a": "algorithm_a_perception_mapping.wav",
    "algorithm_b": "algorithm_b_frequency_shifting.wav",
    "algorithm_c": "algorithm_c_pitch_matching.wav",
    "algorithm_d": "algorithm_d_haptic_gen.wav"
  },
  "events": [
    {
      "event_id": "event_001",
      "category": "gunshot",
      "start_sec": 14.02,
      "peak_sec": 14.18,
      "end_sec": 14.95,
      "included_in_gate": true
    }
  ]
}
```

## Quick start (Colab)

1. Open [`notebooks/haptic_groundtruth_colab.ipynb`](notebooks/haptic_groundtruth_colab.ipynb)
2. **Runtime → Change runtime type → T4 GPU** (recommended)
3. Run all cells (code auto-installs; no project upload)
4. Upload a 3–5 min video
5. Set `GATE_CATEGORIES` in the run cell, review `events.json`, download ZIP

## Local run

```bash
pip install -r requirements.txt
# ffmpeg required

python -c "
from haptic_gt.pipeline import generate_candidate_tracks
tracks = generate_candidate_tracks(
    'sample.mp4', 'output/',
    from_video=True,
    enable_context_detection=True,
    gate_categories=['gunshot', 'explosion'],
)
print(tracks.save_all())
"
```

## Project layout

```
haptic_gt/
  algorithms/          # Sound2Hap A–D
  context/
    taxonomy.yaml
    proposals.py       # RMS onset proposals
    tokenization.py
    context_detectors.py
    encoders.py
    event_aggregation.py
    frozen_fusion.py
    mask.py
    detector.py
  haptic_synthesis.py  # per-event stitch → full timeline
  pipeline.py
```

## Attribution

Algorithms adapted from [Iris1215/Sound2Hap](https://github.com/Iris1215/Sound2Hap) (MIT license).
