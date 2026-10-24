"""AST second opinion on each event's category, at the timing the pipeline found.

PANNs (the default frame-level backend) places events precisely but labels them
less accurately than AST (AudioSet mAP 0.385 vs 0.459). Here AST scores one
window per final event. Start, peak and end are never changed.

Modes (detector param ``ast_verify``):

``off``       nothing runs (default).
``annotate``  each event gets an ``ast`` record: AST's best category, its score,
              AST's score for the event's own category, and a status.
``relabel``   as annotate, and an event whose category AST clearly rejects in
              favour of another of the same kind (impulsive or sustained) takes
              AST's category. A category AST has no labels for (car_crash, which
              only the picture confirms) is never relabelled.

Statuses: ``agree``, ``relabeled``, ``disagree`` (AST prefers another category
but the rule above kept the original), ``unconfirmed`` (no category reached
``ast_verify_min_score``), ``not_scorable`` (the event's category has no AudioSet
labels).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

import numpy as np

from .frozen_fusion import DetectedEvent, dedupe_events_by_peak
from .taxonomy import Taxonomy

MODES = ("off", "annotate", "relabel")
SR = 16_000

#: Per window: {category: (best AudioSet label, score)} for every category AST scores.
WindowScores = dict[str, tuple[str, float]]
Scorer = Callable[[list[np.ndarray], Taxonomy], list[WindowScores]]


def normalize_mode(mode: str | bool | None) -> str:
    if mode is None or mode is False:
        return "off"
    if mode is True:
        return "relabel"
    mode = str(mode).lower()
    if mode not in MODES:
        raise ValueError(f"ast_verify must be one of {MODES}, got {mode!r}")
    return mode


def event_window(ev: DetectedEvent, taxonomy: Taxonomy, duration_sec: float) -> tuple[float, float]:
    """A window centred on the peak, slid inside the clip so it keeps its length.

    Sustained events get the longer window and ignore their span: a rumble's
    span can end at its peak, and a window clipped to it hears only the build-up.
    """
    cat = taxonomy.categories.get(ev.category)
    if cat is not None and cat.impulsive:
        width = taxonomy.ast_verify_window_sec
    else:
        width = taxonomy.ast_verify_sustained_window_sec
    width = min(width, duration_sec)
    t0 = min(max(0.0, ev.peak_sec - width / 2.0), duration_sec - width)
    return t0, t0 + width


def ast_scorer(clips: list[np.ndarray], taxonomy: Taxonomy, *, batch_size: int = 12) -> list[WindowScores]:
    """Score clips with AST: all 527 sigmoid outputs, folded into taxonomy categories."""
    import torch

    from .sed_frames import _category_label_index, _device_str, _get_ast

    model, extractor = _get_ast()
    id2label = model.config.id2label
    cat_idx = _category_label_index(taxonomy, id2label)
    out: list[WindowScores] = []
    for i in range(0, len(clips), batch_size):
        inputs = extractor(
            clips[i : i + batch_size], sampling_rate=SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(_device_str()) for k, v in inputs.items()}
        probs = torch.sigmoid(model(**inputs).logits).detach().cpu().numpy()
        for row in probs:
            scores: WindowScores = {}
            for cat, idxs in cat_idx.items():
                best = max(idxs, key=lambda j: row[j])
                scores[cat] = (str(id2label[best]), float(row[best]))
            out.append(scores)
    return out


def _verdict(ev: DetectedEvent, scores: WindowScores, taxonomy: Taxonomy, mode: str) -> tuple[DetectedEvent, str]:
    own_label, own_score = scores.get(ev.category, ("", 0.0))
    best_cat, (best_label, best_score) = max(
        scores.items(), key=lambda kv: kv[1][1], default=(None, ("", 0.0))
    )
    record = {
        "category": best_cat if best_score >= taxonomy.ast_verify_min_score else None,
        "label": best_label,
        "score": round(best_score, 4),
        "own_score": round(own_score, 4),
    }

    cur = taxonomy.categories.get(ev.category)
    if ev.category not in scores:
        status = "not_scorable"
    elif best_score < taxonomy.ast_verify_min_score:
        status = "unconfirmed"
    elif best_cat == ev.category:
        status = "agree"
    else:
        new = taxonomy.categories.get(best_cat)
        same_kind = cur is not None and new is not None and cur.impulsive == new.impulsive
        clear = best_score >= taxonomy.ast_verify_margin * own_score
        status = "relabeled" if mode == "relabel" and same_kind and clear else "disagree"

    record["status"] = status
    if status == "relabeled":
        record["relabeled_from"] = ev.category
        return replace(ev, category=best_cat, label=best_label, ast=record), status
    return replace(ev, ast=record), status


def verify_events_with_ast(
    events: list[DetectedEvent],
    audio_16k: np.ndarray,
    taxonomy: Taxonomy,
    mode: str,
    *,
    scorer: Scorer | None = None,
) -> tuple[list[DetectedEvent], dict]:
    """Attach AST's verdict to every event and, in ``relabel`` mode, apply it."""
    mode = normalize_mode(mode)
    if mode == "off":
        return events, {"mode": "off"}
    report: dict = {
        "mode": mode,
        "window_sec": taxonomy.ast_verify_window_sec,
        "min_score": taxonomy.ast_verify_min_score,
        "margin": taxonomy.ast_verify_margin,
    }
    if not events:
        return events, {**report, "events": 0}

    duration = len(audio_16k) / float(SR)
    clips = []
    for ev in events:
        t0, t1 = event_window(ev, taxonomy, duration)
        s0 = int(t0 * SR)
        s1 = max(int(t1 * SR), s0 + SR // 10)
        clips.append(audio_16k[s0:s1].astype(np.float32))
    all_scores = (scorer or ast_scorer)(clips, taxonomy)

    verified: list[DetectedEvent] = []
    statuses: Counter[str] = Counter()
    for ev, scores in zip(events, all_scores):
        new_ev, status = _verdict(ev, scores, taxonomy, mode)
        verified.append(new_ev)
        statuses[status] += 1

    if statuses["relabeled"]:
        verified = dedupe_events_by_peak(verified)
    report.update({"events": len(events), **{k: statuses.get(k, 0) for k in (
        "agree", "relabeled", "disagree", "unconfirmed", "not_scorable"
    )}})
    return verified, report
