"""AST category check on fused events (fake scorer, no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from haptic_gt.components.detectors.fusion_detector.ast_verify import (
    context_window,
    event_window,
    normalize_mode,
    verify_events_with_ast,
)
from haptic_gt.components.detectors.fusion_detector.component import Component, to_event
from haptic_gt.components.detectors.fusion_detector.frozen_fusion import DetectedEvent
from haptic_gt.components.detectors.fusion_detector.mask import (
    event_included_in_gate,
    events_for_haptic_gate,
)
from haptic_gt.components.detectors.fusion_detector.taxonomy import load_taxonomy

AUDIO = np.zeros(16_000 * 20, dtype=np.float32)


def _event(category, peak, start=None, end=None, label="x", sources=("audio",)):
    return DetectedEvent(
        category=category,
        label=label,
        start_sec=peak - 0.1 if start is None else start,
        peak_sec=peak,
        end_sec=peak + 0.4 if end is None else end,
        confidence=0.5,
        sources=list(sources),
    )


def _run(events, mode, *windows):
    """``windows``: per event, its short-window scores or (short, context)."""
    pairs = [w if isinstance(w, tuple) else (w, {}) for w in windows]

    def scorer(clips, taxonomy):
        assert len(clips) == 2 * len(events)
        return [s for s, _ in pairs] + [c for _, c in pairs]

    return verify_events_with_ast(events, AUDIO, load_taxonomy(), mode, scorer=scorer)


def test_relabel_within_impulsive_keeps_timing():
    ev = _event("gunshot", 5.0, label="Gunshot, gunfire")
    out, report = _run(
        [ev], "relabel",
        {"gunshot": ("Gunshot, gunfire", 0.2), "explosion": ("Explosion", 0.8)},
    )
    assert out[0].category == "explosion"
    assert out[0].label == "Explosion"
    assert (out[0].start_sec, out[0].peak_sec, out[0].end_sec) == (4.9, 5.0, 5.4)
    assert out[0].ast["status"] == "relabeled"
    assert out[0].ast["relabeled_from"] == "gunshot"
    assert report["relabeled"] == 1


def test_annotate_records_but_never_relabels():
    ev = _event("gunshot", 5.0)
    out, report = _run(
        [ev], "annotate",
        {"gunshot": ("Gunshot, gunfire", 0.2), "explosion": ("Explosion", 0.8)},
    )
    assert out[0].category == "gunshot"
    assert out[0].ast == {
        "category": "explosion",
        "label": "Explosion",
        "score": 0.8,
        "own_score": 0.2,
        "status": "disagree",
    }
    assert report["disagree"] == 1


def test_no_relabel_across_impulsive_and_sustained():
    ev = _event("vehicle", 5.0, start=3.0, end=8.0)
    out, _ = _run(
        [ev], "relabel",
        {"vehicle": ("Engine", 0.1), "explosion": ("Explosion", 0.9)},
    )
    assert out[0].category == "vehicle"
    assert out[0].ast["status"] == "disagree"


def test_no_relabel_without_clear_margin():
    ev = _event("gunshot", 5.0)
    out, _ = _run(
        [ev], "relabel",
        {"gunshot": ("Gunshot, gunfire", 0.5), "explosion": ("Explosion", 0.6)},
    )
    assert out[0].category == "gunshot"
    assert out[0].ast["status"] == "agree"


def test_context_window_confirms_a_shot_under_dialogue():
    ev = _event("gunshot", 5.0)
    out, _ = _run(
        [ev], "relabel",
        ({"gunshot": ("Gunshot, gunfire", 0.08)}, {"gunshot": ("Fusillade", 0.34)}),
    )
    assert out[0].ast["status"] == "agree"
    assert out[0].ast["own_score"] == 0.34
    assert out[0].ast["label"] == "Fusillade"


def test_context_window_never_relabels():
    # A neighbouring explosion inside the 4 s window must not take over this shot
    ev = _event("gunshot", 5.0)
    out, _ = _run(
        [ev], "relabel",
        ({"gunshot": ("Gunshot, gunfire", 0.1)}, {"explosion": ("Explosion", 0.9)}),
    )
    assert out[0].category == "gunshot"
    assert out[0].ast["status"] == "weak"


def test_statuses():
    events = [
        _event("explosion", 2.0),
        _event("smash", 6.0),
        _event("gunshot", 8.0),
        _event("car_crash", 10.0),
    ]
    out, report = _run(
        events, "relabel",
        {"explosion": ("Explosion", 0.7), "gunshot": ("Gunshot, gunfire", 0.1)},
        {"smash": ("Shatter", 0.1), "explosion": ("Explosion", 0.25)},
        {"gunshot": ("Gunshot, gunfire", 0.05)},
        {"smash": ("Smash, crash", 0.9)},
    )
    assert [e.ast["status"] for e in out] == ["agree", "weak", "unconfirmed", "not_scorable"]
    assert out[1].ast["category"] is None
    assert not any(e.excluded_from_gate for e in out)
    assert (report["agree"], report["weak"], report["unconfirmed"], report["not_scorable"]) == (
        1, 1, 1, 1
    )
    assert report["gated_out"] == 0


def test_filter_gates_out_unconfirmed_but_keeps_the_event():
    tax = load_taxonomy()
    quiet = _event("gunshot", 3.0)
    weak = _event("gunshot", 8.0)
    flash = _event("gunshot", 12.0, sources=("audio", "visual_flash"))
    out, report = _run(
        [quiet, weak, flash], "filter",
        ({"gunshot": ("Gunshot, gunfire", 0.04)}, {"gunshot": ("Cap gun", 0.08)}),
        ({"gunshot": ("Gunshot, gunfire", 0.1)}, {"gunshot": ("Fusillade", 0.21)}),
        ({"gunshot": ("Gunshot, gunfire", 0.05)}, {"gunshot": ("Gunshot, gunfire", 0.1)}),
    )
    assert len(out) == 3
    assert out[0].excluded_from_gate and out[0].ast["gated_out"] is True
    assert out[1].ast["status"] == "weak" and not out[1].excluded_from_gate
    assert out[2].ast["kept_by_picture"] is True and not out[2].excluded_from_gate
    assert report["gated_out"] == 1

    gate_cats = ["gunshot"]
    assert events_for_haptic_gate(out, tax, gate_categories=gate_cats) == out[1:]
    assert not event_included_in_gate(out[0], tax, gate_categories=gate_cats)
    assert [to_event(e, tax, gate_cats).in_gate for e in out] == [False, True, True]


def test_relabel_mode_never_gates_out():
    out, report = _run(
        [_event("gunshot", 3.0)], "relabel",
        {"gunshot": ("Gunshot, gunfire", 0.01)},
    )
    assert out[0].ast["status"] == "unconfirmed"
    assert not out[0].excluded_from_gate
    assert report["gated_out"] == 0


def test_off_does_nothing():
    ev = _event("gunshot", 5.0)
    out, report = verify_events_with_ast([ev], AUDIO, load_taxonomy(), "off")
    assert out == [ev] and out[0].ast is None
    assert report == {"mode": "off"}


def test_event_windows():
    tax = load_taxonomy()

    def window(ev, duration=20.0, fn=event_window):
        return pytest.approx(fn(ev, tax, duration))

    assert window(_event("gunshot", 5.0)) == (4.5, 5.5)
    assert window(_event("vehicle", 10.0, start=2.0, end=18.0)) == (9.0, 11.0)
    assert window(_event("vehicle", 10.0, start=9.9, end=10.2)) == (9.0, 11.0)
    # Span ending at its peak (war tank rumble): the window still covers after it
    assert window(_event("vehicle", 2.592, start=0.092, end=2.592)) == (1.592, 3.592)
    # Near the clip edges the window slides inside instead of shrinking
    assert window(_event("vehicle", 0.5, start=0.0, end=2.0)) == (0.0, 2.0)
    assert window(_event("gunshot", 19.9)) == (19.0, 20.0)
    assert window(_event("vehicle", 1.0, start=0.0, end=2.0), duration=1.5) == (0.0, 1.5)
    assert window(_event("gunshot", 5.0), fn=context_window) == (3.0, 7.0)
    assert window(_event("gunshot", 1.0), fn=context_window) == (0.0, 4.0)


def test_modes():
    assert normalize_mode(None) == "off"
    assert normalize_mode(False) == "off"
    assert normalize_mode(True) == "relabel"
    assert normalize_mode("Annotate") == "annotate"
    assert normalize_mode("filter") == "filter"
    with pytest.raises(ValueError):
        normalize_mode("sometimes")
    with pytest.raises(ValueError):
        Component(ast_verify="sometimes")
