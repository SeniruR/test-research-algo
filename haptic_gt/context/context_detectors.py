"""Frozen transformer-based context detectors for sudden symbolic events."""

from __future__ import annotations

from dataclasses import dataclass

from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy, match_label_to_category


@dataclass
class SymbolicToken:
    time_sec: float
    label: str
    confidence: float
    modality: str  # "audio" | "video"
    category: str | None = None


IMPULSIVE_AUDIO_KEYWORDS = (
    "thunder",
    "explosion",
    "gunshot",
    "gunfire",
    "machine gun",
    "fireworks",
    "bang",
    "boom",
    "artillery",
    "fusillade",
    "cap gun",
)

IMPULSIVE_VIDEO_KEYWORDS = (
    "shooting",
    "explod",
    "fire",
    "smash",
    "hit",
    "crash",
)

SUSTAINED_AUDIO_KEYWORDS = (
    "vehicle",
    "engine",
    "truck",
    "motor vehicle",
    "idling",
    "tank",
    "rain",
    "wind",
    "storm",
)

SUSTAINED_VIDEO_KEYWORDS = (
    "driving",
    "motorcycl",
    "riding",
)


def _is_impulsive_label(label: str, keywords: tuple[str, ...]) -> bool:
    low = label.lower()
    return any(k in low for k in keywords)


def symbolic_tokens_from_scores(
    encoder_scores: list[EncoderScore],
    taxonomy: Taxonomy | None = None,
    *,
    threshold: float | None = None,
) -> list[SymbolicToken]:
    """
    Emit symbolic tokens from a single encoder pass.

    Impulsive labels use context_detector_threshold; sustained labels (vehicle,
    weather-non-thunder) use the lower sustained_encoder_threshold so they are
    not silently filtered out.
    """
    taxonomy = taxonomy or load_taxonomy()
    impulsive_thresh = threshold if threshold is not None else taxonomy.context_detector_threshold
    sustained_thresh = taxonomy.sustained_encoder_threshold
    tokens: list[SymbolicToken] = []

    for enc in encoder_scores:
        if enc.source == "audio":
            is_impulsive = _is_impulsive_label(enc.label, IMPULSIVE_AUDIO_KEYWORDS)
            is_sustained = _is_impulsive_label(enc.label, SUSTAINED_AUDIO_KEYWORDS)
            modality = "audio"
        else:
            is_impulsive = _is_impulsive_label(enc.label, IMPULSIVE_VIDEO_KEYWORDS)
            is_sustained = _is_impulsive_label(enc.label, SUSTAINED_VIDEO_KEYWORDS)
            modality = "video"

        if not is_impulsive and not is_sustained:
            continue

        effective_thresh = impulsive_thresh if is_impulsive else sustained_thresh
        if enc.score < effective_thresh:
            continue

        cat = match_label_to_category(taxonomy, enc.label, modality)
        if cat is None:
            continue

        tokens.append(
            SymbolicToken(
                time_sec=enc.time_sec,
                label=enc.label,
                confidence=enc.score,
                modality=modality,
                category=cat,
            )
        )

    return tokens


def detect_symbolic_tokens(
    video_path,
    audio_16k,
    video_frames,
    taxonomy: Taxonomy | None = None,
    *,
    window_sec: float = 0.5,
    hop_sec: float = 0.5,
    duration_sec: float | None = None,
    threshold: float | None = None,
    encoder_scores: list[EncoderScore] | None = None,
) -> list[SymbolicToken]:
    """
    Emit symbolic tokens for sudden AV events.

    When encoder_scores is provided, filters that list (no extra model calls).
    Otherwise falls back to legacy full-timeline classification (deprecated).
    """
    if encoder_scores is not None:
        return symbolic_tokens_from_scores(
            encoder_scores,
            taxonomy=taxonomy,
            threshold=threshold,
        )

    if duration_sec is None:
        raise ValueError("duration_sec is required when encoder_scores is not provided")

    from pathlib import Path

    from haptic_gt.context.encoders import run_encoder_pass

    taxonomy = taxonomy or load_taxonomy()
    scores = run_encoder_pass(
        Path(video_path),
        audio_16k,
        video_frames,
        window_sec=window_sec,
        hop_sec=hop_sec,
        duration_sec=duration_sec,
        full_scan=True,
    )
    return symbolic_tokens_from_scores(scores, taxonomy=taxonomy, threshold=threshold)
