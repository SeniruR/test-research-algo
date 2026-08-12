"""Event-based and segment-based F1, following the DCASE / sed_eval conventions.

Two standard views of the same detections:

Event-based
    A detection matches a reference event when their onsets fall within a collar
    (DCASE Task 4 uses 200 ms) and, optionally, when the offsets agree within
    ``offset_collar_sec`` or a percentage of the reference length. Matching is
    one-to-one, so duplicate detections on one reference count as insertions.

Segment-based
    The timeline is cut into fixed segments (1 s by default) and each segment is
    scored for presence/absence per category. Insensitive to onset jitter, so it
    answers "did we find the event at all" rather than "is the timing tight".

Reference events may be given as full spans, or as onset-only marks (a hand-made
list of times), which is what listening by ear produces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RefEvent:
    category: str
    onset_sec: float
    offset_sec: float | None = None


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    n_ref: int
    n_det: int
    tp: int
    fp: int
    fn: int


def _prf(tp: int, fp: int, fn: int, n_ref: int, n_det: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return PRF(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        n_ref=n_ref,
        n_det=n_det,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def _as_ref_events(
    reference: list[RefEvent] | list[dict] | dict[str, list[float]],
) -> list[RefEvent]:
    """Accept RefEvent list, events.json-style dicts, or {category: [onsets]}."""
    if isinstance(reference, dict):
        out: list[RefEvent] = []
        for cat, times in reference.items():
            for t in times:
                out.append(RefEvent(category=cat, onset_sec=float(t)))
        return sorted(out, key=lambda r: (r.category, r.onset_sec))
    out = []
    for item in reference:
        if isinstance(item, RefEvent):
            out.append(item)
            continue
        onset = item.get("onset_sec", item.get("start_sec", item.get("peak_sec")))
        offset = item.get("offset_sec", item.get("end_sec"))
        out.append(
            RefEvent(
                category=str(item.get("category", "")),
                onset_sec=float(onset),
                offset_sec=None if offset is None else float(offset),
            )
        )
    return sorted(out, key=lambda r: (r.category, r.onset_sec))


def _detection_onsets(
    detections: list[dict],
    *,
    onset_field: str,
) -> list[tuple[str, float, float | None]]:
    rows: list[tuple[str, float, float | None]] = []
    for d in detections:
        cat = str(d.get("category", ""))
        onset = d.get(onset_field)
        if onset is None:
            onset = d.get("start_sec", d.get("peak_sec", 0.0))
        end = d.get("end_sec")
        rows.append((cat, float(onset), None if end is None else float(end)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def event_based_prf(
    reference,
    detections: list[dict],
    *,
    onset_collar_sec: float = 0.2,
    offset_collar_sec: float | None = None,
    offset_collar_rate: float = 0.2,
    categories: list[str] | None = None,
    onset_field: str = "peak_sec",
) -> dict:
    """One-to-one onset matching within a collar, per category and overall.

    ``offset_collar_sec`` None disables offset checking (onset-only scoring),
    which is the right mode for hand-marked onsets.
    """
    refs = _as_ref_events(reference)
    dets = _detection_onsets(detections, onset_field=onset_field)
    cats = categories or sorted({r.category for r in refs} | {d[0] for d in dets})

    per_cat: dict[str, PRF] = {}
    matches: list[dict] = []
    tot_tp = tot_fp = tot_fn = 0
    tot_ref = tot_det = 0

    for cat in cats:
        cat_refs = [r for r in refs if r.category == cat]
        cat_dets = [d for d in dets if d[0] == cat]
        used_det: set[int] = set()
        tp = 0
        for ref in cat_refs:
            best: tuple[float, int] | None = None
            for j, (_, onset, end) in enumerate(cat_dets):
                if j in used_det:
                    continue
                err = abs(onset - ref.onset_sec)
                if err > onset_collar_sec:
                    continue
                if (
                    offset_collar_sec is not None
                    and ref.offset_sec is not None
                    and end is not None
                ):
                    tol = max(
                        offset_collar_sec,
                        offset_collar_rate * (ref.offset_sec - ref.onset_sec),
                    )
                    if abs(end - ref.offset_sec) > tol:
                        continue
                if best is None or err < best[0]:
                    best = (err, j)
            if best is None:
                matches.append(
                    {
                        "category": cat,
                        "ref_onset_sec": round(ref.onset_sec, 3),
                        "det_onset_sec": None,
                        "error_sec": None,
                        "status": "miss",
                    }
                )
                continue
            err, j = best
            used_det.add(j)
            tp += 1
            matches.append(
                {
                    "category": cat,
                    "ref_onset_sec": round(ref.onset_sec, 3),
                    "det_onset_sec": round(cat_dets[j][1], 3),
                    "error_sec": round(cat_dets[j][1] - ref.onset_sec, 3),
                    "status": "hit",
                }
            )
        for j, (_, onset, _end) in enumerate(cat_dets):
            if j in used_det:
                continue
            matches.append(
                {
                    "category": cat,
                    "ref_onset_sec": None,
                    "det_onset_sec": round(onset, 3),
                    "error_sec": None,
                    "status": "false_positive",
                }
            )
        fp = len(cat_dets) - tp
        fn = len(cat_refs) - tp
        per_cat[cat] = _prf(tp, fp, fn, len(cat_refs), len(cat_dets))
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_ref += len(cat_refs)
        tot_det += len(cat_dets)

    micro = _prf(tot_tp, tot_fp, tot_fn, tot_ref, tot_det)
    macro_f1 = (
        round(sum(v.f1 for v in per_cat.values()) / len(per_cat), 4) if per_cat else 0.0
    )
    errors = [abs(m["error_sec"]) for m in matches if m["error_sec"] is not None]
    return {
        "mode": "event_based",
        "onset_collar_sec": onset_collar_sec,
        "offset_collar_sec": offset_collar_sec,
        "micro": asdict(micro),
        "macro_f1": macro_f1,
        "per_category": {k: asdict(v) for k, v in per_cat.items()},
        "mean_abs_onset_error_sec": (
            round(sum(errors) / len(errors), 4) if errors else None
        ),
        "matches": sorted(
            matches, key=lambda m: (m["ref_onset_sec"] or m["det_onset_sec"] or 0.0)
        ),
    }


def segment_based_prf(
    reference,
    detections: list[dict],
    *,
    duration_sec: float,
    segment_sec: float = 1.0,
    default_ref_length_sec: float = 0.5,
    categories: list[str] | None = None,
) -> dict:
    """Presence/absence per fixed segment, per category and overall.

    Onset-only references are given ``default_ref_length_sec`` so they occupy a
    segment at all.
    """
    refs = _as_ref_events(reference)
    cats = categories or sorted(
        {r.category for r in refs} | {str(d.get("category", "")) for d in detections}
    )
    n_seg = max(1, int(round(duration_sec / segment_sec)))

    def _active(spans: list[tuple[float, float]], seg: int) -> bool:
        s0 = seg * segment_sec
        s1 = s0 + segment_sec
        return any(start < s1 and end > s0 for start, end in spans)

    per_cat: dict[str, PRF] = {}
    tot_tp = tot_fp = tot_fn = 0
    for cat in cats:
        ref_spans = [
            (
                r.onset_sec,
                r.offset_sec
                if r.offset_sec is not None
                else r.onset_sec + default_ref_length_sec,
            )
            for r in refs
            if r.category == cat
        ]
        det_spans = [
            (
                float(d.get("start_sec", d.get("peak_sec", 0.0))),
                float(d.get("end_sec", d.get("peak_sec", 0.0))),
            )
            for d in detections
            if str(d.get("category", "")) == cat
        ]
        tp = fp = fn = 0
        for seg in range(n_seg):
            r_on = _active(ref_spans, seg)
            d_on = _active(det_spans, seg)
            if r_on and d_on:
                tp += 1
            elif d_on:
                fp += 1
            elif r_on:
                fn += 1
        per_cat[cat] = _prf(tp, fp, fn, tp + fn, tp + fp)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    micro = _prf(tot_tp, tot_fp, tot_fn, tot_tp + tot_fn, tot_tp + tot_fp)
    macro_f1 = (
        round(sum(v.f1 for v in per_cat.values()) / len(per_cat), 4) if per_cat else 0.0
    )
    return {
        "mode": "segment_based",
        "segment_sec": segment_sec,
        "micro": asdict(micro),
        "macro_f1": macro_f1,
        "per_category": {k: asdict(v) for k, v in per_cat.items()},
    }


def coverage_report(
    reference,
    detections: list[dict],
    *,
    categories: list[str],
    duration_sec: float | None = None,
    tolerance_sec: float = 0.35,
) -> dict:
    """Recall-only view: does a detected span contain each mark?

    Sustained rumble cannot be scored by onset collar or by segment presence when
    the reference is a handful of "I feel it here" marks: the marks are moments
    inside a burst, not onsets, and they say nothing about the seconds between
    them. Precision needs full reference spans, so only recall is reported, next
    to how much of the clip was marked active so over-triggering stays visible.

    ``tolerance_sec`` widens each span, because a mark typed as "6" is a rounded
    second and the burst it refers to may start at 6.3. Uncovered marks carry
    their distance to the nearest span so a boundary rounding is not mistaken for
    a silent stretch.
    """
    refs = _as_ref_events(reference)
    per_cat: dict[str, dict] = {}
    for cat in categories:
        marks = [r.onset_sec for r in refs if r.category == cat]
        spans = [
            (
                float(d.get("start_sec", d.get("peak_sec", 0.0))),
                float(d.get("end_sec", d.get("peak_sec", 0.0))),
            )
            for d in detections
            if str(d.get("category", "")) == cat
        ]

        def _distance(m: float) -> float:
            if not spans:
                return float("inf")
            return min(max(s - m, m - e, 0.0) for s, e in spans)

        distances = {m: _distance(m) for m in marks}
        covered = sum(1 for m in marks if distances[m] <= tolerance_sec)
        active = sum(max(0.0, e - s) for s, e in spans)
        per_cat[cat] = {
            "n_marks": len(marks),
            "covered": covered,
            "recall": round(covered / len(marks), 4) if marks else 0.0,
            "n_spans": len(spans),
            "active_sec": round(active, 3),
            "active_fraction": (
                round(active / duration_sec, 4)
                if duration_sec and duration_sec > 0
                else None
            ),
            "tolerance_sec": tolerance_sec,
            "uncovered_marks": [
                {
                    "mark_sec": round(m, 3),
                    "distance_to_span_sec": (
                        None if distances[m] == float("inf") else round(distances[m], 3)
                    ),
                }
                for m in marks
                if distances[m] > tolerance_sec
            ],
        }
    return {"mode": "coverage", "per_category": per_cat}


def evaluate_events(
    reference,
    events_json: dict | list[dict],
    *,
    duration_sec: float,
    collars_sec: tuple[float, ...] = (0.2, 0.5),
    segment_sec: float = 1.0,
    onset_field: str = "peak_sec",
    sustained_categories: tuple[str, ...] = (),
    sustained_tolerance_sec: float = 0.35,
) -> dict:
    """Event-based F1 at several collars, segment-based F1, sustained coverage.

    Categories in ``sustained_categories`` are held out of the F1 blocks and get
    the coverage view instead; see ``coverage_report``.
    """
    detections = (
        events_json.get("events", [])
        if isinstance(events_json, dict)
        else list(events_json)
    )
    sustained = set(sustained_categories)
    refs = _as_ref_events(reference)
    onset_refs = [r for r in refs if r.category not in sustained]
    onset_dets = [d for d in detections if str(d.get("category", "")) not in sustained]

    report = {
        "event_based": [
            event_based_prf(
                onset_refs,
                onset_dets,
                onset_collar_sec=c,
                onset_field=onset_field,
            )
            for c in collars_sec
        ],
        "segment_based": segment_based_prf(
            onset_refs,
            onset_dets,
            duration_sec=duration_sec,
            segment_sec=segment_sec,
        ),
    }
    if sustained:
        report["coverage"] = coverage_report(
            refs,
            detections,
            categories=sorted(sustained & {r.category for r in refs}),
            duration_sec=duration_sec,
            tolerance_sec=sustained_tolerance_sec,
        )
    return report


def _fmt_prf(name: str, prf: dict) -> str:
    return (
        name.rjust(16)
        + format(prf["precision"], ".3f").rjust(11)
        + format(prf["recall"], ".3f").rjust(9)
        + format(prf["f1"], ".3f").rjust(8)
        + str(prf["n_ref"]).rjust(7)
        + str(prf["n_det"]).rjust(7)
        + str(prf["tp"]).rjust(5)
        + str(prf["fp"]).rjust(5)
        + str(prf["fn"]).rjust(5)
    )


def format_evaluation(report: dict) -> str:
    header = (
        "        category  precision   recall      F1  n_ref  n_det   TP   FP   FN"
    )
    lines: list[str] = []
    scored = any(
        block["micro"]["n_ref"] or block["micro"]["n_det"]
        for block in report["event_based"]
    )
    if not scored:
        lines.append(
            "Event-based / segment-based F1: no onset references or detections in "
            "this clip, nothing to score (zeros below would be meaningless)."
        )
    for block in report["event_based"] if scored else []:
        collar = block["onset_collar_sec"]
        mean_err = block["mean_abs_onset_error_sec"]
        lines.append(
            "Event-based F1 (onset collar {0} s){1}".format(
                collar,
                ""
                if mean_err is None
                else "  |  mean |onset error| = {0:.3f} s".format(mean_err),
            )
        )
        lines.append(header)
        for cat, prf in block["per_category"].items():
            lines.append(_fmt_prf(cat, prf))
        lines.append(_fmt_prf("MICRO", block["micro"]))
        lines.append("      macro F1: {0:.3f}".format(block["macro_f1"]))
        lines.append("")

    if scored:
        seg = report["segment_based"]
        lines.append("Segment-based F1 ({0} s segments)".format(seg["segment_sec"]))
        lines.append(header)
        for cat, prf in seg["per_category"].items():
            lines.append(_fmt_prf(cat, prf))
        lines.append(_fmt_prf("MICRO", seg["micro"]))
        lines.append("      macro F1: {0:.3f}".format(seg["macro_f1"]))

    cov = report.get("coverage")
    if cov and cov["per_category"]:
        lines.append("")
        lines.append("Sustained coverage (recall only; sparse marks cannot score precision)")
        lines.append("        category   covered   recall  spans  active_s  active_%")
        for cat, row in cov["per_category"].items():
            frac = row["active_fraction"]
            lines.append(
                cat.rjust(16)
                + "{0}/{1}".format(row["covered"], row["n_marks"]).rjust(10)
                + format(row["recall"], ".3f").rjust(9)
                + str(row["n_spans"]).rjust(7)
                + format(row["active_sec"], ".1f").rjust(10)
                + ("-" if frac is None else format(100.0 * frac, ".1f")).rjust(10)
            )
            if row["uncovered_marks"]:
                lines.append(
                    "                  missed marks: "
                    + ", ".join(
                        "{0:.2f} ({1})".format(
                            m["mark_sec"],
                            "no span"
                            if m["distance_to_span_sec"] is None
                            else "{0:.2f}s away".format(m["distance_to_span_sec"]),
                        )
                        for m in row["uncovered_marks"]
                    )
                )

    misses = [
        m
        for block in report["event_based"][:1]
        for m in block["matches"]
        if m["status"] != "hit"
    ]
    if misses:
        lines.append("")
        lines.append("Unmatched at tightest collar:")
        for m in misses:
            if m["status"] == "miss":
                lines.append(
                    "  miss  {0:>12}  ref {1:.3f}".format(
                        m["category"], m["ref_onset_sec"]
                    )
                )
            else:
                lines.append(
                    "  FP    {0:>12}  det {1:.3f}".format(
                        m["category"], m["det_onset_sec"]
                    )
                )
    return "\n".join(lines)
