"""Merge symbolic tokens into event primitives with start/peak/end."""

from __future__ import annotations

from dataclasses import dataclass, field

from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.context_detectors import SymbolicToken
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy, match_label_to_category


@dataclass
class EventPrimitive:
    category: str | None
    label: str
    start_sec: float
    peak_sec: float
    end_sec: float
    confidence: float
    sources: list[str] = field(default_factory=list)
    context_token: bool = False
    audio_score: float | None = None
    video_score: float | None = None


def _merge_tokens(
    items: list[SymbolicToken],
    taxonomy: Taxonomy,
) -> list[EventPrimitive]:
    """Peak-split impulsive tokens; gap-merge sustained ones."""
    if not items:
        return []

    by_category: dict[str, list[SymbolicToken]] = {}
    for tok in items:
        cat = tok.category or "unknown"
        by_category.setdefault(cat, []).append(tok)

    primitives: list[EventPrimitive] = []
    for cat_name, group in by_category.items():
        cat_cfg = taxonomy.categories.get(cat_name)
        if cat_cfg and cat_cfg.impulsive:
            scores = [
                EncoderScore(
                    time_sec=t.time_sec,
                    label=t.label,
                    score=t.confidence,
                    source=t.modality,
                )
                for t in group
            ]
            primitives.extend(
                _peak_split_scores(
                    scores,
                    category=cat_name,
                    taxonomy=taxonomy,
                    context_token=True,
                )
            )
            continue

        sorted_items = sorted(group, key=lambda x: x.time_sec)
        merge_gap = taxonomy.sustained_merge_gap_sec
        clusters: list[list[SymbolicToken]] = [[sorted_items[0]]]
        for tok in sorted_items[1:]:
            if tok.time_sec - clusters[-1][-1].time_sec <= merge_gap:
                clusters[-1].append(tok)
            else:
                clusters.append([tok])
        for cluster in clusters:
            peak_tok = max(cluster, key=lambda x: x.confidence)
            half_win = 0.35
            primitives.append(
                EventPrimitive(
                    category=cat_name if cat_name != "unknown" else None,
                    label=peak_tok.label,
                    start_sec=max(0.0, cluster[0].time_sec - half_win),
                    peak_sec=peak_tok.time_sec,
                    end_sec=cluster[-1].time_sec + half_win,
                    confidence=peak_tok.confidence,
                    sources=sorted({g.modality for g in cluster}),
                    context_token=True,
                )
            )
    return primitives


def _local_peaks(
    times: list[float],
    scores: list[float],
    *,
    min_score: float,
    min_distance_sec: float,
) -> list[int]:
    """Indices of local maxima separated by at least min_distance_sec."""
    if not times:
        return []

    # Collapse duplicate timestamps to max score
    buckets: dict[float, float] = {}
    for t, s in zip(times, scores):
        buckets[t] = max(buckets.get(t, 0.0), s)
    ordered_t = sorted(buckets)
    ordered_s = [buckets[t] for t in ordered_t]

    candidates: list[int] = []
    n = len(ordered_t)
    for i in range(n):
        if ordered_s[i] < min_score:
            continue
        left = ordered_s[i - 1] if i > 0 else -1.0
        right = ordered_s[i + 1] if i + 1 < n else -1.0
        if ordered_s[i] >= left and ordered_s[i] >= right:
            candidates.append(i)

    # Greedy keep highest peaks first, enforce spacing
    candidates.sort(key=lambda i: ordered_s[i], reverse=True)
    kept: list[int] = []
    for i in candidates:
        if all(abs(ordered_t[i] - ordered_t[j]) >= min_distance_sec for j in kept):
            kept.append(i)
    kept.sort()
    return kept


def _peak_split_scores(
    scores: list[EncoderScore],
    *,
    category: str,
    taxonomy: Taxonomy,
    context_token: bool = False,
) -> list[EventPrimitive]:
    """One short event per local score peak (for gunshots / explosions)."""
    if not scores:
        return []

    min_score = taxonomy.impulsive_encoder_threshold
    half = taxonomy.impulsive_event_half_width_sec
    min_dist = taxonomy.impulsive_min_peak_distance_sec

    times = [s.time_sec for s in scores]
    vals = [s.score for s in scores]
    peak_idxs = _local_peaks(
        times, vals, min_score=min_score, min_distance_sec=min_dist
    )

    buckets: dict[float, float] = {}
    for t, s in zip(times, vals):
        buckets[t] = max(buckets.get(t, 0.0), s)
    ordered_t = sorted(buckets)

    if not peak_idxs:
        best = max(scores, key=lambda s: s.score)
        if best.score < min_score:
            return []
        peak_times = {best.time_sec}
    else:
        peak_times = {ordered_t[i] for i in peak_idxs}

    peak_scores: list[EncoderScore] = []
    for t in sorted(peak_times):
        nearby = [s for s in scores if abs(s.time_sec - t) <= 0.01]
        if not nearby:
            continue
        peak_scores.append(max(nearby, key=lambda s: s.score))

    primitives: list[EventPrimitive] = []
    for peak in peak_scores:
        nearby = [
            s
            for s in scores
            if abs(s.time_sec - peak.time_sec) <= half + 0.15
            and s.score >= min_score * 0.8
        ]
        sources = sorted({s.source for s in (nearby or [peak])})
        audio_scores = [s.score for s in nearby if s.source == "audio"] or (
            [peak.score] if peak.source == "audio" else []
        )
        video_scores = [s.score for s in nearby if s.source == "video"] or (
            [peak.score] if peak.source == "video" else []
        )
        primitives.append(
            EventPrimitive(
                category=category,
                label=peak.label,
                start_sec=max(0.0, peak.time_sec - half),
                peak_sec=peak.time_sec,
                end_sec=peak.time_sec + half,
                confidence=peak.score,
                sources=sources,
                context_token=context_token,
                audio_score=max(audio_scores) if audio_scores else None,
                video_score=max(video_scores) if video_scores else None,
            )
        )
    return primitives


def _merge_encoder_scores(
    scores: list[EncoderScore],
    taxonomy: Taxonomy,
    *,
    gap_sec: float | None = None,
) -> list[EventPrimitive]:
    """Aggregate encoder scores by taxonomy category."""
    if not scores:
        return []

    gap = gap_sec if gap_sec is not None else taxonomy.sustained_merge_gap_sec
    by_category: dict[str, list[EncoderScore]] = {}
    for s in scores:
        cat = match_label_to_category(taxonomy, s.label, s.source)
        if cat is None:
            continue
        by_category.setdefault(cat, []).append(s)

    primitives: list[EventPrimitive] = []
    for cat_name, group in by_category.items():
        cat_cfg = taxonomy.categories.get(cat_name)
        if cat_cfg and cat_cfg.impulsive:
            primitives.extend(
                _peak_split_scores(group, category=cat_name, taxonomy=taxonomy)
            )
            continue

        # Sustained categories (vehicle, weather): gap-merge above-threshold hits
        threshold = getattr(taxonomy, "sustained_encoder_threshold", taxonomy.encoder_threshold)
        group = sorted(
            [s for s in group if s.score >= threshold],
            key=lambda x: x.time_sec,
        )
        if not group:
            continue
        clusters: list[list[EncoderScore]] = [[group[0]]]
        for item in group[1:]:
            if item.time_sec - clusters[-1][-1].time_sec <= gap:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster in clusters:
            peak = max(cluster, key=lambda x: x.score)
            sources = sorted({c.source for c in cluster})
            audio_scores = [c.score for c in cluster if c.source == "audio"]
            video_scores = [c.score for c in cluster if c.source == "video"]
            primitives.append(
                EventPrimitive(
                    category=cat_name,
                    label=peak.label,
                    start_sec=max(0.0, cluster[0].time_sec - 0.25),
                    peak_sec=peak.time_sec,
                    end_sec=cluster[-1].time_sec + 0.25,
                    confidence=peak.score,
                    sources=sources,
                    context_token=False,
                    audio_score=max(audio_scores) if audio_scores else None,
                    video_score=max(video_scores) if video_scores else None,
                )
            )
    return primitives


def _overlaps(a: EventPrimitive, b: EventPrimitive, *, margin_sec: float = 0.35) -> bool:
    if a.category != b.category:
        return False
    # Impulsive: only treat as same event if peaks are very close
    if abs(a.peak_sec - b.peak_sec) <= margin_sec:
        return True
    return not (a.end_sec + margin_sec < b.start_sec or b.end_sec + margin_sec < a.start_sec)


def aggregate_events(
    tokens: list[SymbolicToken],
    encoder_scores: list[EncoderScore],
    taxonomy: Taxonomy | None = None,
) -> list[EventPrimitive]:
    """Combine context symbolic tokens and encoder scores into primitives."""
    taxonomy = taxonomy or load_taxonomy()
    token_primitives = _merge_tokens(tokens, taxonomy)
    encoder_primitives = _merge_encoder_scores(encoder_scores, taxonomy)

    if not token_primitives:
        return encoder_primitives
    if not encoder_primitives:
        return token_primitives

    merged = list(token_primitives)
    for enc in encoder_primitives:
        if any(_overlaps(enc, tok) for tok in token_primitives):
            continue
        merged.append(enc)
    merged.sort(key=lambda p: p.start_sec)
    return merged
