# Haptic Ground Truth

Generate **candidate haptic tracks** from short video audio (3–5 min) using the same four signal-processing algorithms as [Sound2Hap](https://github.com/Iris1215/Sound2Hap) (CHI 2026).

Designed for **Google Colab (CPU)** — no GPU required.

## Alignment with Sound2Hap

This project ports the reference implementations from Sound2Hap's `Signal_Processing_Algorithms/` folder:

| Output file | Sound2Hap module | Paper |
|-------------|------------------|-------|
| `algorithm_a_perception_mapping.wav` | `Percept.py` | Lee & Choi, CHI 2013 |
| `algorithm_b_frequency_shifting.wav` | `FreqShift.py` | Okazaki et al., 2015 |
| `algorithm_c_pitch_matching.wav` | `Pitch_WebTool.py` | Kim et al., IEEE ToH 2023 |
| `algorithm_d_haptic_gen.wav` | `HapticGen.py` | Sung et al., CHI 2025 |

Key conventions (matching Sound2Hap):

- **Input:** mono 44.1 kHz audio (extracted from video)
- **Output:** mono **8 kHz** haptic WAV (16-bit PCM)
- **Normalization:** peak / RMS rules from Sound2Hap `utils/normalization.py`
- **Pitch Match:** ISO 532-1 loudness via [MoSQITo](https://github.com/Eomys/MoSQITo) (Python web-tool version)

Interactive reference tool: [sound2hap.netlify.app](https://sound2hap.netlify.app/)

## Pipeline

```
Video → Extract Audio (44.1 kHz mono)
     → Algorithm A (Perceptual Mapping)
     → Algorithm B (Frequency Shifting)
     → Algorithm C (Pitch Matching)
     → Algorithm D (HapticGen)
     → 8 kHz candidate WAVs → Human rating → Ground truth dataset
```

## Quick start (Colab)

1. Open [`notebooks/haptic_groundtruth_colab.ipynb`](notebooks/haptic_groundtruth_colab.ipynb) in Google Colab.
2. **Runtime → Change runtime type → CPU**
3. Run all cells — code auto-installs in cell 2 (no project upload).
4. Upload a 3–5 min video when prompted.
5. Download the ZIP of candidate WAV files.

## Local run (optional)

```bash
pip install -r requirements.txt
# ffmpeg must be on PATH

python -c "
from haptic_gt.pipeline import generate_candidate_tracks
tracks = generate_candidate_tracks('sample.mp4', 'output/', from_video=True, content_type='game')
print(tracks.save_all())
"
```

## Project layout

```
haptic-groundtruth/
├── haptic_gt/
│   ├── algorithms/
│   │   ├── percept.py       # Algorithm A
│   │   ├── freq_shift.py    # Algorithm B
│   │   ├── pitch_match.py   # Algorithm C
│   │   └── haptic_gen.py    # Algorithm D
│   ├── utils/
│   │   └── normalization.py
│   ├── audio_io.py
│   └── pipeline.py
├── notebooks/
│   └── haptic_groundtruth_colab.ipynb
└── requirements.txt
```

## Attribution

Algorithms adapted from [Iris1215/Sound2Hap](https://github.com/Iris1215/Sound2Hap) (MIT license). If you use this in research, cite Sound2Hap:

```bibtex
@article{li2026sound2hap,
    title={Sound2Hap: Learning Audio-to-Vibrotactile Haptic Generation from Human Ratings},
    author={Li, Yinan and Seifi, Hasti},
    journal={arXiv preprint arXiv:2601.12245},
    year={2026}
}
```
