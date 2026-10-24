"""AST second opinion on each event's category, at the timing the pipeline found.

PANNs (the default frame-level backend) places events precisely but labels them
less accurately than AST (AudioSet mAP 0.385 vs 0.459). Here AST scores two
windows around each final event's peak: a short one (what this sound is) and a
wider context one (whether this kind of sound is here at all; a shot under
dialogue or music only shows up with more context). Start, peak and end are
never changed.

Modes (detector param ``ast_verify``):

``off``       nothing runs (default).
``annotate``  each event gets an ``ast`` record: AST's best category and label,
              its score, AST's score for the event's own category, and a status.
``relabel``   as annotate, and an event whose category AST clearly rejects in
              favour of another of the same kind (impulsive or sustained) takes
              AST's category.
``filter``    as relabel, and an event AST hears none of our categories in is
              taken out of the haptic gate. It stays in events.json.

Statuses:

``agree``         AST hears the event's category (``ast_verify_min_score``).
``relabeled``     the short window clearly favours another category of the same kind.
``disagree``      AST hears another category, but the relabel rule kept this one.
``weak``          AST hears something of ours, below ``ast_verify_min_score``
                  but at least ``ast_verify_filter_score``. Kept.
``unconfirmed``   below ``ast_verify_filter_score`` in both windows. ``filter``
                  gates it out unless the picture backs it (orange flash or a
                  scene label).
``not_scorable``  the category has no AudioSet labels (car_crash, which only the
                  picture confirms). Never changed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

import numpy as np

from .frozen_fusion import DetectedEvent, dedupe_events_by_peak
from .taxonomy import Taxonomy

MODES = ("off", "annotate", "relabel", "filter")
STATUSES = ("agree", "relabeled", "disagree", "weak", "unconfirmed", "not_scorable")
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


def _centred(peak_sec: float, width: float, duration_sec: float) -> tuple[float, float]:
    width = min(width, duration_sec)
    t0 = min(max(0.0, peak_sec - width / 2.0), duration_sec - width)
    return t0, t0 + width


def event_window(ev: DetectedEvent, taxonomy: Taxonomy, duration_sec: float) -> tuple[float, float]:
    """The short window: centred on the peak, slid inside the clip so it keeps its length.

    Sustained events get the longer window and ignore their span: a rumble's
    span can end at its peak, and a window clipped to it hears only the build-up.
    """
    cat = taxonomy.categories.get(ev.category)
    if cat is not None and cat.impulsive:
        width = taxonomy.ast_verify_window_sec
    else:
        width = taxonomy.ast_verify_sustained_window_sec
    return _centred(ev.peak_sec, width, duration_sec)


def context_window(ev: DetectedEvent, taxonomy: Taxonomy, duration_sec: float) -> tuple[float, float]:
    return _centred(ev.peak_sec, taxonomy.ast_verify_context_sec, duration_sec)


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


def _best(scores: WindowScores, exclude: str | None = None) -> tuple[str | None, str, float]:
    cat, (label, score) = max(
        ((c, v) for c, v in scores.items() if c != exclude),
        key=lambda kv: kv[1][1],
        default=(None, ("", 0.0)),
    )
    return cat, label, score


def _verdict(
    ev: DetectedEvent,
    short: WindowScores,
    context: WindowScores,
    taxonomy: Taxonomy,
    mode: str,
) -> tuple[DetectedEvent, str]:
    # Either window may confirm a category; only the short one may relabel, so a
    # neighbouring sound inside the context window cannot steal the event.
    heard = {
        c: max(short.get(c, ("", 0.0)), context.get(c, ("", 0.0)), key=lambda v: v[1])
        for c in set(short) | set(context)
    }
    own = heard.get(ev.category, ("", 0.0))[1]
    heard_cat, heard_label, heard_score = _best(heard)
    rival_cat, rival_label, rival_score = _best(short, exclude=ev.category)
    min_score = taxonomy.ast_verify_min_score

    cur = taxonomy.categories.get(ev.category)
    new = taxonomy.categories.get(rival_cat) if rival_cat else None
    same_kind = cur is not None and new is not None and cur.impulsive == new.impulsive
    if ev.category not in heard:
        status = "not_scorable"
    elif (
        mode in ("relabel", "filter")
        and same_kind
        and rival_score >= min_score
        and rival_score >= taxonomy.ast_verify_margin * own
    ):
        status = "relabeled"
    elif own >= min_score:
        status = "agree"
    elif rival_score >= min_score:
        status = "disagree"
    elif heard_score >= taxonomy.ast_verify_filter_score:
        status = "weak"
    else:
        status = "unconfirmed"

    record: dict = {
        "category": heard_cat if heard_score >= min_score else None,
        "label": heard_label,
        "score": round(heard_score, 4),
        "own_score": round(own, 4),
        "status": status,
    }
    if status == "relabeled":
        record["relabeled_from"] = ev.category
        return replace(ev, category=rival_cat, label=rival_label, ast=record), status
    if status == "unconfirmed" and mode == "filter":
        if "visual_flash" in ev.sources or ev.visual_categories:
            record["kept_by_picture"] = True
        else:
            record["gated_out"] = True
            return replace(ev, ast=record, excluded_from_gate=True), status
    return replace(ev, ast=record), status


def verify_events_with_ast(
    events: list[DetectedEvent],
    audio_16k: np.ndarray,
    taxonomy: Taxonomy,
    mode: str,
    *,
    scorer: Scorer | None = None,
) -> tuple[list[DetectedEvent], dict]:
    """Attach AST's verdict to every event and apply it as ``mode`` says."""
    mode = normalize_mode(mode)
    if mode == "off":
        return events, {"mode": "off"}
    report: dict = {
        "mode": mode,
        "window_sec": taxonomy.ast_verify_window_sec,
        "sustained_window_sec": taxonomy.ast_verify_sustained_window_sec,
        "context_sec": taxonomy.ast_verify_context_sec,
        "min_score": taxonomy.ast_verify_min_score,
        "filter_score": taxonomy.ast_verify_filter_score,
        "margin": taxonomy.ast_verify_margin,
    }
    if not events:
        return events, {**report, "events": 0}

    duration = len(audio_16k) / float(SR)

    def clip(window: tuple[float, float]) -> np.ndarray:
        s0 = int(window[0] * SR)
        s1 = max(int(window[1] * SR), s0 + SR // 10)
        return audio_16k[s0:s1].astype(np.float32)

    clips = [clip(event_window(ev, taxonomy, duration)) for ev in events]
    clips += [clip(context_window(ev, taxonomy, duration)) for ev in events]
    scores = (scorer or ast_scorer)(clips, taxonomy)
    n = len(events)

    verified: list[DetectedEvent] = []
    statuses: Counter[str] = Counter()
    for ev, short, context in zip(events, scores[:n], scores[n:]):
        new_ev, status = _verdict(ev, short, context, taxonomy, mode)
        verified.append(new_ev)
        statuses[status] += 1

    if statuses["relabeled"]:
        verified = dedupe_events_by_peak(verified)
    report.update({"events": n, **{s: statuses.get(s, 0) for s in STATUSES}})
    report["gated_out"] = sum(1 for e in verified if e.excluded_from_gate)
    return verified, report
