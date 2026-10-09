# Haptic Ground Truth

Turns a video into **candidate 8 kHz vibration tracks** for human-in-the-loop
comparison on the Android app (`vibrator-android`). A context detector finds the
events worth feeling; haptic generators — the four
[Sound2Hap](https://github.com/Iris1215/Sound2Hap) algorithms plus a rule-based
mapper — render one WAV each.

The pipeline is modular. Components are folders: add one and list it in
`pipeline.yaml`, or remove its line (and folder). Nothing in the core changes.

## Architecture

```
video ──► core: extract source_audio.wav (44.1 kHz mono)
            │
            ▼
      context detector (one)          components/detectors/<name>/
            │  events with start / peak / end,
            │  impulsive?  in the haptic gate?
            ▼
      core: gated_audio.wav + events.json
            │
            ▼
      haptic generators (any number)  components/generators/<name>/
        input "events":   one clip per gated event, stitched onto the timeline by the core
        input "full_mix": the whole ungated source, once
            │
            ▼
      algorithm_<slot>_<name>.wav (8 kHz) ──► ZIP for the Android app
```

```
pipeline.yaml                 which components run, with what params
haptic_gt/
  core/                       fixed: never edited to add or remove a component
    contracts.py              Event, DetectionResult, GeneratorContext, the two Protocols
    config.py                 loads pipeline.yaml (+ per-run overrides)
    registry.py               finds component folders, builds them, validates slots
    runner.py                 run_pipeline(): extract → detect → gate → generate
    audio_io.py gating.py synthesis.py packaging.py vendor.py
  components/
    detectors/
      fusion_detector/        audio (PANNs/AST) + video (flashes, Qwen2.5-VL) fusion; taxonomy.yaml
      manual_events/          hand-labelled events instead of detection; categories.yaml
    generators/
      a_perception_mapping/   Sound2Hap Percept.py
      b_frequency_shifting/   Sound2Hap FreqShift.py
      c_pitch_matching/       Sound2Hap Pitch_WebTool.py
      d_haptic_gen/           Sound2Hap HapticGen.py
      e_rule_based/           rule-based thunder / rain / RMS mapper (full mix + video)
third_party/sound2hap/        unmodified upstream Sound2Hap files + SHA256SUMS
notebooks/haptic_groundtruth_colab.ipynb
```

The core knows no categories. A detector owns its taxonomy and marks each event
`impulsive` (an accent placed at the peak) or sustained (follows its span), and
whether it is `in_gate`. Components import only `haptic_gt.core`, their own
folder and `third_party/`; `tests/test_components.py` enforces this, so deleting
one component folder cannot break another.

## pipeline.yaml

```yaml
detector:
  name: fusion_detector
  params: { use_qwen: null, gate_categories: null, manual_rumble_peaks: null }

generators:
  - name: a_perception_mapping
    params: { content: game }
  - name: b_frequency_shifting
  - name: c_pitch_matching
  - name: d_haptic_gen
  - name: e_rule_based

synthesis:
  continuous: { enabled: false }   # true adds a low-level bed between accents
  sustained_mask_min_events: 3
  sustained_mask_min_coverage: 0.15
```

`detector: null` skips detection; then only `full_mix` generators (E) produce
output. Overrides apply per run without editing the file:

```python
from haptic_gt import load_pipeline_config, run_pipeline

config = load_pipeline_config(overrides={
    "detector": {"name": "manual_events", "params": {"events": "marks.json"}},
    "generators": ["a_perception_mapping", "e_rule_based"],
})
result = run_pipeline(config, "clip.mp4", "output/")
print(result.save_all())
```

A detector override with a different `name` replaces the configured detector
(and its params); the same name merges params. `generators` is replaced as a list.

## Adding or removing a component

**Remove:** delete its line from `pipeline.yaml`. Delete the folder too if you
like; nothing else refers to it.

**Add a generator:** create `haptic_gt/components/generators/<name>/component.py`:

```python
from pathlib import Path
from haptic_gt.core.contracts import GeneratorContext

class Component:
    slot = "b"                                  # Android compare slot, a–e
    output_name = "algorithm_b_my_method.wav"
    input = "events"                            # or "full_mix"

    def __init__(self, strength: float = 1.0):  # params from pipeline.yaml
        self.strength = strength

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext):
        ...  # write a mono WAV to out_wav: any rate for "events" (the core
             # resamples), ctx.output_sr (8 kHz) for "full_mix"
        return {"anything": "worth logging"}    # optional, lands in run reports
```

then list `- name: <name>` under `generators:`.

**Add a detector:** create `haptic_gt/components/detectors/<name>/component.py`
with a `Component` whose `detect(video_path, source_wav, workdir)` returns a
`DetectionResult` of `Event`s (set `impulsive` and `in_gate` on each). Put
anything you want in `report`; it is merged into `events.json`. Then set
`detector: {name: <name>}`.

Unknown names fail with the list of available components; bad params, duplicate
slots and duplicate output names fail before anything runs.

### Android app slots

The app loads files named `algorithm_{a..e}_*.wav` from the ZIP root
(`HapticFolderMatcher.kt`), so at most **five generators** run at once, each with
a distinct `slot` letter, and `output_name` must follow that pattern. To compare a
new method, give it the slot of one you drop.

## Sound2Hap: verifying the algorithms are unmodified

Generators A–D call the original upstream scripts in `third_party/sound2hap/`,
byte-identical to [Iris1215/Sound2Hap](https://github.com/Iris1215/Sound2Hap) at
the commit in `third_party/sound2hap/UPSTREAM.md` (MIT, see its `LICENSE`).
Check them any of these ways:

```bash
python scripts/verify_sound2hap.py              # files vs SHA256SUMS
python scripts/verify_sound2hap.py --upstream   # also vs GitHub at the pinned commit
cd third_party/sound2hap && sha256sum -c SHA256SUMS
```

or delete the `.py` files and drop in your own download of the upstream
`Signal_Processing_Algorithms/` files (`--download` does this). The Colab notebook
runs the check after cloning, and `.gitattributes` keeps git from rewriting their
line endings.

**Pitch matching (C) and MoSQITo.** Upstream `Pitch_WebTool.py` imports
`mosqito.functions.loudness_zwtv._loudness_zwtv` and reads a dict result. No
MoSQITo release or commit ever had that module (0.x: `functions/loudness_zwicker`;
1.x: `sq_metrics.loudness_zwtv`, returning a tuple), so upstream always uses its
mean-envelope loudness fallback and C sits near 50 Hz. That is the default here,
matching what the published script produces. `c_pitch_matching: {iso532_loudness:
true}` feeds the unmodified script real ISO 532-1 loudness from MoSQITo 1.x via
`c_pitch_matching/mosqito_adapter.py`; it costs roughly 7 s of CPU per second of
event audio. The generator report
says which was used (`"loudness": "envelope_fallback"` or `"iso532_1"`).

## Fusion detector

Categories, gate defaults and every threshold live in
`haptic_gt/components/detectors/fusion_detector/taxonomy.yaml`:

| Category | Kind | In gate by default |
|----------|------|--------------------|
| `gunshot` | impulsive | yes |
| `explosion` | impulsive | yes |
| `smash` | impulsive | yes |
| `car_crash` | impulsive | yes |
| `vehicle` | sustained (rumble) | yes |

Audio branch: framewise sound-event detection with PANNs (`panns-inference`,
~10 ms frames) or dense AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) as
fallback, sharpened with spectral-flux onsets. Video branch: orange
muzzle/explosion flash scan, plus optional Qwen2.5-VL scene labels. With
`use_qwen: null` (the default) Qwen runs only on a GPU with at least
`visual_scenes_min_gpu_gb` (20 GB) of memory, so it is skipped on a T4 or CPU
and runs on an L4 or A100; `true` or `false` forces it. The choice and the reason
are printed and recorded under `detector` in `events.json`. Params: `gate_categories`, `use_qwen`, `full_scan`,
`manual_rumble_peaks` (replace auto vehicle spans with hand-marked times),
`taxonomy_path`.

## manual_events detector

Trusts hand-labelled times. `events` is a path to JSON (a list, `{"events":
[...]}` such as a previous `events.json`, or one event) or the same inline. Each
event needs `category` and `peak_sec`; missing `start_sec` / `end_sec` default
from `categories.yaml`.

## Outputs

| File | From |
|------|------|
| `algorithm_a_perception_mapping.wav` … `algorithm_e_rule_based.wav` | generators, 8 kHz mono, full clip length |
| `events.json` | detector events + report, `haptic_outputs`, `no_haptic_events` |
| `gated_audio.wav` | source with everything outside gated events silenced |
| `source_audio.wav` | 44.1 kHz mono extraction |
| `debug/<component>/` | auxiliary files a component wrote |

Event-driven generators are skipped when no event is in the gate
(`no_haptic_events: true`); `full_mix` generators still run.

`core.packaging.write_candidate_archive` builds the app ZIP: the video and
generator WAVs at the root, `events.json` and `gated_audio.wav` under
`debug/context_detector/`, source audio under `debug/audio_preparation/`, and
everything else under `debug/`.

## Quick start (Colab)

1. Open [`notebooks/haptic_groundtruth_colab.ipynb`](notebooks/haptic_groundtruth_colab.ipynb)
   in Colab; **Runtime → Change runtime type → T4 GPU**.
2. For a private repo, add a `GITHUB_TOKEN` secret (key icon).
3. Run the cells in order: clone + verify Sound2Hap, install, fetch the PANNs
   model (downloaded once from Zenodo, MD5-checked, then cached in your Google
   Drive at `MyDrive/haptic-groundtruth/models/`), choose
   components (`OVERRIDES`), upload a video, run, inspect the timeline, listen,
   download the ZIP.

## Local run

```bash
pip install -r requirements.txt   # plus ffmpeg on PATH
python -c "from haptic_gt import run_pipeline; print(run_pipeline(None, 'clip.mp4', 'output/').save_all())"
```

`run_pipeline(None, ...)` uses `pipeline.yaml` as is.

## Tests and regression check

```bash
python -m pytest tests
python scripts/regression_baseline.py check   # outputs vs tests/baseline hashes
```

The baseline renders the reference tank clip with `manual_events` (no ML, so
deterministic) and compares the SHA-256 of every WAV plus the event timings.
After an intended output change, re-record with `regression_baseline.py record`.

## Attribution

Algorithms A–D: [Iris1215/Sound2Hap](https://github.com/Iris1215/Sound2Hap) (MIT
license), vendored unmodified in `third_party/sound2hap/`.
