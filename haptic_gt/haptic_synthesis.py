"""Stitch per-event haptic segments onto a full-length timeline."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d

from haptic_gt.audio_io import INPUT_SR, VIB_SR
from haptic_gt.context.frozen_fusion import DetectedEvent
from haptic_gt.context.mask import build_event_mask
from haptic_gt.context.sustained_merge import sustained_coverage_sec
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy


def _impulsive_accent_events(
    events: list[DetectedEvent],
    taxonomy: Taxonomy,
) -> list[DetectedEvent]:
    """Only impulsive categories get bang-style accents (gunshot/explosion/thunder)."""
    out: list[DetectedEvent] = []
    for ev in events:
        cat = taxonomy.categories.get(ev.category)
        if cat is not None and cat.impulsive:
            out.append(ev)
    return out


def _sustained_events(
    events: list[DetectedEvent],
    taxonomy: Taxonomy,
) -> list[DetectedEvent]:
    out: list[DetectedEvent] = []
    for ev in events:
        cat = taxonomy.categories.get(ev.category)
        if cat is not None and not cat.impulsive:
            out.append(ev)
    return out


def _use_sustained_bed_mask(
    sustained: list[DetectedEvent],
    duration_sec: float,
    taxonomy: Taxonomy,
) -> bool:
    """True for intermittent rumble clips; false for a single short vehicle chip."""
    if not sustained or duration_sec <= 0:
        return False
    n = len(sustained)
    if n >= taxonomy.sustained_mask_min_events:
        return True
    # One sparse detection (e.g. short tank idle chip) keeps the full bed
    if n < 2:
        return False
    coverage = sustained_coverage_sec(sustained, taxonomy) / duration_sec
    return coverage >= taxonomy.sustained_mask_min_coverage


@dataclass(frozen=True)
class ContinuousProfile:
    """
    Shape of the continuous haptic layer rendered between detected events.

    Defaults are tuned against the rule-based reference map
    (`sample/videoplayback_output_haptic_map[1].json`), which keeps ~71% of its
    40 ms windows active at a median intensity of 50/255 while reserving the top
    of the range for event peaks.
    """

    enabled: bool = True

    #: Level the continuous layer's loud passages sit at, measured as the 95th
    #: percentile of its envelope rather than its absolute peak so the bed has a
    #: predictable strength regardless of how spiky the algorithm output is.
    base_gain: float = 0.24
    #: Envelope percentile that `base_gain` refers to.
    base_level_pct: float = 0.95

    # -- Input levelling -----------------------------------------------------
    # Algorithms A-D peak-normalize whatever they are handed and apply their own
    # perceptual thresholds, so a quiet passage inside a loud clip renders as
    # silence (A returns literal zeros below its detection threshold). Levelling
    # the audio first keeps every active passage above those thresholds.
    #: Exponent applied to the source envelope when levelling. Lower is flatter.
    agc_strength: float = 0.05
    #: Window used when measuring the source envelope.
    agc_window_ms: float = 50.0
    #: Peak-hold applied to the envelope before deriving gain, so transients are
    #: not amplified along with the quiet material around them.
    agc_hold_ms: float = 150.0
    #: Ceiling on levelling gain, in dB.
    agc_max_gain_db: float = 40.0

    # -- Output dynamics -----------------------------------------------------
    #: Exponent re-imposing macro dynamics on the levelled output, so louder
    #: passages still feel stronger. 1.0 restores the source's full range.
    #: A rumble is one long span whose bursts and lulls are all the haptic has to
    #: convey its rhythm with, so this has to stay well clear of a flat bed: at
    #: 0.30 a burst measured 2.9x its gap came out 0.98x, i.e. no rhythm at all.
    dynamics_strength: float = 0.65
    #: Fraction of the quietest frames muted, so real silence stays silent.
    #: The reference map leaves 29% of its windows empty.
    silence_pct: float = 0.28
    #: Frames louder than this fraction of the peak are never muted, so a clip
    #: that is active end to end is not silenced just to hit `silence_pct`.
    max_floor_ratio: float = 0.15

    # -- Event accents -------------------------------------------------------
    #: Continuous layer is attenuated to this fraction underneath an event so
    #: the accent keeps its full dynamic punch without clipping the sum.
    event_duck: float = 0.28
    #: Ramp applied at the edges of each duck region.
    duck_ramp_ms: float = 30.0
    #: Peak amplitude each event segment is normalized to.
    event_gain: float = 0.92
    #: Accents are placed this far ahead of the audio attack. Event peaks land
    #: within ~10 ms of the attack and the rendered accent within ~10 ms of the
    #: event peak, yet the vibration still feels late: a low-frequency actuator
    #: needs a few cycles to be felt, so a haptic aligned to the sample is felt
    #: after the sound. Leading it by a fraction of that rise time lines the two
    #: up perceptually.
    accent_lead_ms: float = 25.0
    #: Release applied where one accent would still be ringing under the next, so
    #: a volley reads as separate hits instead of one long buzz.
    accent_release_ms: float = 45.0
    #: Peak amplitude the loudest sustained rumble span is normalized to.
    sustained_soft_gain: float = 0.48
    #: Quieter rumble spans are scaled by their own loudness relative to the
    #: loudest one, so a distant drive does not hit as hard as a car alongside.
    #: 1.0 would be literal; some compression keeps the quiet scene perceptible.
    sustained_level_exp: float = 0.6
    #: Floor on that scaling, so the quietest rumble is still felt.
    sustained_level_floor: float = 0.35


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def _extract_clip_wav(
    source_wav: Path,
    start_sec: float,
    end_sec: float,
    dest: Path,
    *,
    sample_rate: int = INPUT_SR,
) -> None:
    audio, sr = _read_mono(source_wav)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    s0 = max(0, int(start_sec * sr))
    s1 = min(len(audio), int(end_sec * sr))
    _write_wav(dest, audio[s0:s1], sr)


def _event_clip_bounds(
    ev: DetectedEvent,
    duration_sec: float,
    *,
    min_clip_sec: float = 0.6,
) -> tuple[float, float]:
    """Bounds that include the onset peak with enough post-event context."""
    start = max(0.0, min(ev.start_sec, ev.peak_sec - 0.05))
    end = max(ev.end_sec, ev.peak_sec + 0.35)
    if end - start < min_clip_sec:
        end = min(duration_sec, start + min_clip_sec)
        start = max(0.0, end - min_clip_sec)
    end = min(duration_sec, end)
    return start, end


def _haptic_onset_sec(segment: np.ndarray, sample_rate: int, *, ratio: float = 0.18) -> float:
    """First time the haptic |signal| crosses a fraction of its peak (attack)."""
    if segment.size == 0:
        return 0.0
    abs_seg = np.abs(segment)
    peak = float(np.max(abs_seg))
    if peak < 1e-10:
        return 0.0
    thr = peak * ratio
    hits = np.where(abs_seg >= thr)[0]
    if hits.size == 0:
        return float(np.argmax(abs_seg) / sample_rate)
    return float(hits[0] / sample_rate)


def _normalize_segment(segment: np.ndarray, *, target: float = 0.85) -> np.ndarray:
    """Scale segment so its peak lands at `target` (haptic actuator headroom)."""
    peak = float(np.max(np.abs(segment)))
    if peak < 1e-10:
        return segment
    return segment * (target / peak)


def _scale_by_energy(
    segment: np.ndarray,
    *,
    target: float,
    crest: float = 0.707,
    ceiling: float = 0.95,
) -> np.ndarray:
    """Level a rumble span by its energy instead of its loudest spike.

    Peak normalizing a span makes one holding a sharp transient come out quiet
    and a smooth one come out loud, which puts the power on the wrong span: a
    distant drive rendered as a steady tone then feels stronger than a car
    alongside whose burst has an attack in it. ``crest`` is the peak/RMS ratio of
    a sine, so a smooth rumble still lands on ``target``; spikier material is
    only pulled back if it would run out of headroom.
    """
    rms = float(np.sqrt(np.mean(np.square(segment))))
    peak = float(np.max(np.abs(segment)))
    if rms < 1e-10 or peak < 1e-10:
        return segment
    scale = crest * target / rms
    if peak * scale > ceiling:
        scale = ceiling / peak
    return segment * scale


def _release_tail(segment: np.ndarray, keep: int, release: int) -> np.ndarray:
    """Cut a segment to ``keep`` samples with a cosine release, not a click."""
    if keep <= 0 or segment.size <= keep:
        return segment
    out = segment[:keep].copy()
    r = min(release, keep)
    if r > 1:
        out[-r:] *= ((1.0 + np.cos(np.linspace(0.0, np.pi, r))) / 2.0).astype(out.dtype)
    return out


def _source_levels(
    audio: np.ndarray,
    sample_rate: int,
    events: list[DetectedEvent],
) -> list[float]:
    """RMS of the source under each event span."""
    levels: list[float] = []
    for ev in events:
        s0 = max(0, int(ev.start_sec * sample_rate))
        s1 = min(len(audio), int(ev.end_sec * sample_rate))
        seg = audio[s0:s1]
        levels.append(float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0)
    return levels


def _frame_envelope(
    signal: np.ndarray,
    sample_rate: int,
    window_ms: float,
    hold_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Short-time RMS envelope with frame centres in seconds.

    `hold_ms` widens each frame to the loudest value nearby. Deriving gain from
    a peak-held envelope keeps a transient from dragging up the gain applied to
    the quiet material next to it.
    """
    frame = max(1, int(sample_rate * window_ms / 1000.0))
    n_frames = max(1, int(np.ceil(signal.size / frame)))
    padded = np.pad(signal, (0, n_frames * frame - signal.size))
    envelope = np.sqrt(np.mean(np.square(padded.reshape(n_frames, frame)), axis=1))

    hold_radius = int(round(hold_ms / max(window_ms, 1e-6)))
    if hold_radius > 0 and envelope.size > 1:
        envelope = maximum_filter1d(envelope, size=2 * hold_radius + 1, mode="nearest")

    centers_sec = (np.arange(n_frames, dtype=np.float64) * frame + frame / 2.0) / sample_rate
    return envelope, centers_sec


def _resample_curve(
    values: np.ndarray,
    centers_sec: np.ndarray,
    n_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Interpolate a per-frame curve onto a sample timeline at `sample_rate`."""
    if values.size == 1:
        return np.full(n_samples, values[0], dtype=np.float32)
    positions = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.interp(positions, centers_sec, values).astype(np.float32)


def _level_audio(audio: np.ndarray, sample_rate: int, profile: ContinuousProfile) -> np.ndarray:
    """Flatten the source envelope so every active passage drives the algorithm."""
    envelope, centers = _frame_envelope(
        audio, sample_rate, profile.agc_window_ms, profile.agc_hold_ms
    )
    peak = float(np.max(envelope))
    if peak < 1e-10:
        return audio

    normalized = np.maximum(envelope / peak, 1e-6)
    max_gain = float(10.0 ** (profile.agc_max_gain_db / 20.0))
    gain = np.clip(np.power(normalized, profile.agc_strength - 1.0), 0.0, max_gain)
    return audio * _resample_curve(gain, centers, audio.size, sample_rate)


def _dynamics_curve(
    source_audio: np.ndarray,
    source_sr: int,
    profile: ContinuousProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame gain that re-imposes compressed source dynamics, plus a gate.

    Both are derived from the source audio rather than the algorithm output, so
    the continuous layer rises and falls with what is actually audible.
    """
    envelope, centers = _frame_envelope(source_audio, source_sr, profile.agc_window_ms)
    peak = float(np.max(envelope))
    if peak < 1e-10:
        return np.zeros_like(envelope), centers

    normalized = envelope / peak
    threshold = min(
        float(np.quantile(normalized, np.clip(profile.silence_pct, 0.0, 0.9))),
        profile.max_floor_ratio,
    )
    gate = (normalized > threshold).astype(np.float64)
    shape = np.power(np.maximum(normalized, 1e-6), profile.dynamics_strength)
    return shape * gate, centers


def _duck_envelope(
    total_samples: int,
    spans: list[tuple[int, int]],
    sample_rate: int,
    *,
    duck: float,
    ramp_ms: float,
) -> np.ndarray:
    """Gain envelope that dips to `duck` across each span with cosine edges."""
    envelope = np.ones(total_samples, dtype=np.float32)
    if not spans:
        return envelope

    ramp_len = max(1, int(sample_rate * ramp_ms / 1000.0))
    ramp = (1.0 - np.cos(np.linspace(0.0, np.pi, ramp_len))) / 2.0

    for start, end in spans:
        start = max(0, start)
        end = min(total_samples, end)
        if end <= start:
            continue
        envelope[start:end] = np.minimum(envelope[start:end], duck)

        lead_start = max(0, start - ramp_len)
        lead_len = start - lead_start
        if lead_len > 0:
            fade = 1.0 - (1.0 - duck) * ramp[-lead_len:]
            envelope[lead_start:start] = np.minimum(envelope[lead_start:start], fade)

        tail_end = min(total_samples, end + ramp_len)
        tail_len = tail_end - end
        if tail_len > 0:
            fade = 1.0 - (1.0 - duck) * ramp[::-1][:tail_len]
            envelope[end:tail_end] = np.minimum(envelope[end:tail_end], fade)

    return envelope


def _run_algorithm(
    process_file: Callable[..., object],
    clip_in: Path,
    clip_out: Path,
    *,
    output_sr: int,
    process_kwargs: dict,
) -> np.ndarray:
    process_file(clip_in, clip_out, **process_kwargs)
    segment, seg_sr = _read_mono(clip_out)
    if seg_sr != output_sr:
        import librosa

        segment = librosa.resample(segment, orig_sr=seg_sr, target_sr=output_sr)
    return segment.astype(np.float32)


def _render_continuous_layer(
    source_audio: np.ndarray,
    source_sr: int,
    process_file: Callable[..., object],
    work_dir: Path,
    *,
    output_sr: int,
    total_samples: int,
    process_kwargs: dict,
    profile: ContinuousProfile,
    dynamics: np.ndarray,
) -> np.ndarray:
    """Render the whole clip through an algorithm as a low-level bed."""
    levelled = _level_audio(source_audio, source_sr, profile)
    levelled = _normalize_segment(levelled, target=0.95)

    clip_in = work_dir / "base_in.wav"
    clip_out = work_dir / "base_out.wav"
    _write_wav(clip_in, levelled, source_sr)
    base = _run_algorithm(
        process_file, clip_in, clip_out, output_sr=output_sr, process_kwargs=process_kwargs
    )

    if base.size < total_samples:
        base = np.pad(base, (0, total_samples - base.size))
    base = base[:total_samples]

    base = base * dynamics

    envelope, _ = _frame_envelope(base, output_sr, 20.0)
    active = envelope[envelope > 1e-6]
    if active.size == 0:
        return np.zeros_like(base)
    reference = float(np.quantile(active, profile.base_level_pct))
    if reference < 1e-9:
        return _normalize_segment(base, target=profile.base_gain)
    return np.clip(base * (profile.base_gain / reference), -1.0, 1.0)


def stitch_algorithm_output(
    source_wav: Path,
    events: list[DetectedEvent],
    output_path: Path,
    process_file: Callable[..., object],
    *,
    input_sr: int = INPUT_SR,
    output_sr: int = VIB_SR,
    process_kwargs: dict | None = None,
    continuous: ContinuousProfile | None = None,
    taxonomy: Taxonomy | None = None,
) -> None:
    """
    Run an algorithm over the whole clip and mix per-event accents on top.

    Two layers are produced. The continuous layer renders the entire source
    audio so sustained content still produces motion. Impulsive events
    (gunshot / explosion / thunder) add stronger accents aligned to
    ``peak_sec``, with the bed ducked underneath.

    Sustained categories (e.g. vehicle):
    - Sparse / short detection → full continuous bed (steady rumble clips).
    - Many / high-coverage spans → bed masked to those spans + soft
      time-aligned rumble segments (intermittent car rumble), no bang duck.
    """
    process_kwargs = process_kwargs or {}
    profile = continuous or ContinuousProfile()
    taxonomy = taxonomy or load_taxonomy()

    audio, sr = _read_mono(source_wav)
    if sr != input_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=input_sr)
        sr = input_sr

    duration_sec = len(audio) / sr
    total_samples = max(1, int(round(duration_sec * output_sr)))
    timeline = np.zeros(total_samples, dtype=np.float32)

    sustained = _sustained_events(events, taxonomy)
    intermittent = profile.enabled and _use_sustained_bed_mask(
        sustained, duration_sec, taxonomy
    )

    # With continuous bed on: bang accents only for impulsive hits.
    # With continuous off: keep legacy behaviour (accent every gated event).
    if profile.enabled:
        accent_events = _impulsive_accent_events(events, taxonomy)
    else:
        accent_events = list(events)

    if not accent_events and not profile.enabled and not intermittent:
        _write_wav(output_path, timeline, output_sr)
        return

    # One dynamics curve for the whole clip, shared by the bed and the rumble
    # segments so both rise and fall with what is audible.
    dyn_values, dyn_centers = _dynamics_curve(audio, sr, profile)
    dynamics = _resample_curve(dyn_values, dyn_centers, total_samples, output_sr)

    soft_placements: list[tuple[int, np.ndarray]] = []
    placements: list[tuple[int, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for i, ev in enumerate(accent_events):
            clip_start, clip_end = _event_clip_bounds(ev, duration_sec)
            clip_in = tmp_path / f"event_{i}_in.wav"
            clip_out = tmp_path / f"event_{i}_out.wav"
            _extract_clip_wav(source_wav, clip_start, clip_end, clip_in, sample_rate=sr)
            segment = _run_algorithm(
                process_file,
                clip_in,
                clip_out,
                output_sr=output_sr,
                process_kwargs=process_kwargs,
            )
            segment = _normalize_segment(segment, target=profile.event_gain)

            onset_sec = _haptic_onset_sec(segment, output_sr)
            lead_sec = profile.accent_lead_ms / 1000.0
            start_idx = int(round((ev.peak_sec - onset_sec - lead_sec) * output_sr))
            if start_idx < 0:
                segment = segment[-start_idx:]
                start_idx = 0
            if start_idx >= total_samples or segment.size == 0:
                continue
            placements.append((start_idx, segment))

        # A bang still ringing when the next one lands buries its attack, which
        # reads as the next hit arriving late rather than as one loud volley.
        placements.sort(key=lambda p: p[0])
        release = max(1, int(output_sr * profile.accent_release_ms / 1000.0))
        for i in range(len(placements) - 1):
            start, segment = placements[i]
            keep = placements[i + 1][0] - start
            if 0 < keep < segment.size:
                placements[i] = (start, _release_tail(segment, keep, release))

        if intermittent:
            levels = _source_levels(audio, sr, sustained)
            loudest = max(levels) if levels else 0.0
            for i, ev in enumerate(sustained):
                span = max(0.05, ev.end_sec - ev.start_sec)
                clip_in = tmp_path / f"sustained_{i}_in.wav"
                clip_out = tmp_path / f"sustained_{i}_out.wav"
                _extract_clip_wav(
                    source_wav, ev.start_sec, ev.end_sec, clip_in, sample_rate=sr
                )
                segment = _run_algorithm(
                    process_file,
                    clip_in,
                    clip_out,
                    output_sr=output_sr,
                    process_kwargs=process_kwargs,
                )
                # Normalizing every span to the same peak makes a distant drive hit
                # as hard as a car alongside, which is the rumble's power gone.
                relative = 1.0
                if loudest > 1e-9:
                    relative = (levels[i] / loudest) ** profile.sustained_level_exp
                relative = float(np.clip(relative, profile.sustained_level_floor, 1.0))
                segment = _scale_by_energy(
                    segment, target=profile.sustained_soft_gain * relative
                )
                target_len = max(1, int(round(span * output_sr)))
                if segment.size < target_len:
                    segment = np.pad(segment, (0, target_len - segment.size))
                elif segment.size > target_len:
                    segment = segment[:target_len]
                start_idx = max(0, int(round(ev.start_sec * output_sr)))
                if start_idx >= total_samples or segment.size == 0:
                    continue
                # A rumble span is minutes of one event; its bursts and lulls are
                # all the rhythm the haptic has. Algorithm A normalizes every frame
                # to a constant level internally, so without this the whole span
                # comes out as one flat buzz.
                shape = dynamics[start_idx : start_idx + segment.size]
                if shape.size < segment.size:
                    shape = np.pad(shape, (0, segment.size - shape.size), mode="edge")
                peak = float(np.max(shape)) if shape.size else 0.0
                if peak > 1e-6:
                    segment = segment * (shape / peak)
                soft_placements.append((start_idx, segment))

        if profile.enabled:
            base = _render_continuous_layer(
                audio,
                sr,
                process_file,
                tmp_path,
                output_sr=output_sr,
                total_samples=total_samples,
                process_kwargs=process_kwargs,
                profile=profile,
                dynamics=dynamics,
            )
            if intermittent:
                # Rumble on only where sustained events fired; quiet gaps stay quiet
                mask = build_event_mask(
                    total_samples, output_sr, sustained, fade_ms=40.0
                )
                base *= mask
            spans = [(start, start + len(seg)) for start, seg in placements]
            base *= _duck_envelope(
                total_samples,
                spans,
                output_sr,
                duck=profile.event_duck,
                ramp_ms=profile.duck_ramp_ms,
            )
            timeline += base

    for start_idx, segment in soft_placements:
        end_idx = min(total_samples, start_idx + len(segment))
        seg_len = end_idx - start_idx
        if seg_len > 0:
            timeline[start_idx:end_idx] += segment[:seg_len]

    for start_idx, segment in placements:
        end_idx = min(total_samples, start_idx + len(segment))
        seg_len = end_idx - start_idx
        if seg_len > 0:
            timeline[start_idx:end_idx] += segment[:seg_len]

    np.clip(timeline, -1.0, 1.0, out=timeline)
    _write_wav(output_path, timeline, output_sr)
