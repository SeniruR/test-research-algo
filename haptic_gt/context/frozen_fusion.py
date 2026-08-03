"""Phase 2 frozen cross-modal fusion (rule-based, no training)."""

from __future__ import annotations

from dataclasses import dataclass, field

from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.event_aggregation import EventPrimitive, aggregate_events
from haptic_gt.context.context_detectors import SymbolicToken
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy, match_label_to_category


@dataclass
class DetectedEvent:
    category: str
    label: str
    start_sec: float
    peak_sec: float
    end_sec: float
    confidence: float
    context_token: bool = False
    audio_score: float | None = None
    video_score: float | None = None
    sources: list[str] = field(default_factory=list)


def _best_encoder_scores(
    scores: list[EncoderScore],
    taxonomy: Taxonomy,
    center_sec: float,
    tolerance_sec: float = 0.75,
) -> dict[str, tuple[str, float, str]]:
    """Return best audio/video score per category near center_sec."""
    best: dict[str, tuple[str, float, str]] = {}
    for s in scores:
        if abs(s.time_sec - center_sec) > tolerance_sec:
            continue
        cat = match_label_to_category(taxonomy, s.label, s.source)
        if cat is None:
            continue
        prev = best.get(cat)
        if prev is None or s.score > prev[1]:
            best[cat] = (s.label, s.score, s.source)
    return best


def fuse_events(
    tokens: list[SymbolicToken],
    encoder_scores: list[EncoderScore],
    taxonomy: Taxonomy | None = None,
) -> list[DetectedEvent]:
    """
    Frozen cross-modal fusion at event level.

    1. Aggregate tokens/scores into primitives
    2. Map labels to taxonomy categories
    3. Apply per-category fusion rules and thresholds
    """
    taxonomy = taxonomy or load_taxonomy()
    primitives = aggregate_events(tokens, encoder_scores, taxonomy)
    detected: list[DetectedEvent] = []

    for prim in primitives:
        cat_name = prim.category
        if cat_name is None:
            cat_name = match_label_to_category(taxonomy, prim.label, "audio")
        if cat_name is None:
            cat_name = match_label_to_category(taxonomy, prim.label, "video")
        if cat_name is None or cat_name not in taxonomy.categories:
            continue

        cat_cfg = taxonomy.categories[cat_name]
        nearby = _best_encoder_scores(encoder_scores, taxonomy, prim.peak_sec)
        audio_score = prim.audio_score
        video_score = prim.video_score
        if cat_name in nearby:
            lbl, sc, src = nearby[cat_name]
            if src == "audio":
                audio_score = max(audio_score or 0.0, sc)
            else:
                video_score = max(video_score or 0.0, sc)
            if prim.label == lbl or prim.label in lbl:
                pass
            elif audio_score is None and video_score is None:
                prim.label = lbl

        has_context = prim.context_token
        if cat_cfg.impulsive:
            cat_threshold = taxonomy.impulsive_encoder_threshold
        elif hasattr(taxonomy, "sustained_encoder_threshold"):
            cat_threshold = taxonomy.sustained_encoder_threshold
        else:
            cat_threshold = taxonomy.encoder_threshold
        has_audio = (audio_score or 0.0) >= cat_threshold
        has_video = (video_score or 0.0) >= cat_threshold

        if cat_cfg.require_context_or_both:
            passes = has_context or (has_audio and has_video)
            if not passes and has_audio and cat_cfg.impulsive:
                passes = (audio_score or 0.0) >= cat_threshold
            if not passes:
                continue
        else:
            weighted = (
                cat_cfg.audio_weight * (audio_score or 0.0)
                + cat_cfg.video_weight * (video_score or 0.0)
            )
            score_floor = max(prim.confidence, weighted)
            if not has_context and score_floor < cat_threshold:
                continue

        if has_context and (has_audio or has_video):
            prim.confidence = min(1.0, prim.confidence * 1.05)

        detected.append(
            DetectedEvent(
                category=cat_name,
                label=prim.label,
                start_sec=prim.start_sec,
                peak_sec=prim.peak_sec,
                end_sec=prim.end_sec,
                confidence=prim.confidence,
                context_token=prim.context_token,
                audio_score=audio_score,
                video_score=video_score,
                sources=prim.sources,
            )
        )

    # Deduplicate near-duplicate peaks only (keep distinct shots/blasts)
    detected.sort(key=lambda e: (e.category, e.peak_sec, -e.confidence))
    merged: list[DetectedEvent] = []
    for ev in detected:
        if merged and ev.category == merged[-1].category:
            same_peak = abs(ev.peak_sec - merged[-1].peak_sec) <= 0.35
            if same_peak:
                if ev.confidence > merged[-1].confidence:
                    merged[-1] = ev
                continue
        merged.append(ev)

    merged.sort(key=lambda e: e.start_sec)
    return merged
