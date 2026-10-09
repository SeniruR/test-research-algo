"""Manual events: trust hand-labelled start / peak / end times instead of detecting.

``events`` may be a path to a JSON file (a list of events, ``{"events": [...]}``
such as a previous events.json, or one event object), or the same structures
inline. Each event needs ``category`` and ``peak_sec``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from haptic_gt.core.contracts import DetectionResult, Event

CATEGORIES_PATH = Path(__file__).with_name("categories.yaml")


def _event_rows(spec: Any) -> list[dict[str, Any]]:
    if isinstance(spec, (str, Path)):
        payload = json.loads(Path(spec).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("events"), list):
                return list(payload["events"])
            if "category" in payload and "peak_sec" in payload:
                return [payload]
        raise ValueError(f"Unsupported manual events JSON shape: {spec}")
    if isinstance(spec, dict):
        if isinstance(spec.get("events"), list):
            return list(spec["events"])
        return [spec]
    return list(spec or [])


class Component:
    def __init__(
        self,
        events: Any = None,
        gate_categories: list[str] | None = None,
        categories_path: str | None = None,
    ):
        """
        ``gate_categories``: categories that drive the haptics (default: those
        with ``in_gate`` in categories.yaml).
        """
        if events is None:
            raise ValueError("manual_events needs an 'events' param (path, list, or dict)")
        self.events = events
        self.gate_categories = gate_categories
        raw = yaml.safe_load(Path(categories_path or CATEGORIES_PATH).read_text(encoding="utf-8"))
        self.categories: dict[str, dict[str, Any]] = raw.get("categories", {})
        defaults = raw.get("defaults", {})
        self.pre_roll_sec = float(defaults.get("pre_roll_sec", 0.08))
        self.half_width_sec = float(defaults.get("half_width_sec", 0.45))

    def _gate_categories(self) -> list[str]:
        if self.gate_categories is not None:
            return list(self.gate_categories)
        return [name for name, cfg in self.categories.items() if cfg.get("in_gate")]

    def _event(self, row: dict[str, Any], gate_cats: list[str]) -> Event:
        category = str(row["category"]).strip()
        if category not in self.categories:
            raise ValueError(
                f"Unknown category {category!r}. Expected one of: {sorted(self.categories)}"
            )
        peak = float(row["peak_sec"])
        start = (
            float(row["start_sec"])
            if row.get("start_sec") is not None
            else max(0.0, peak - self.pre_roll_sec)
        )
        end = float(row["end_sec"]) if row.get("end_sec") is not None else peak + self.half_width_sec
        if end < start:
            raise ValueError(f"end_sec ({end}) must be >= start_sec ({start})")
        return Event(
            category=category,
            label=str(row.get("label") or category),
            start_sec=start,
            peak_sec=min(max(peak, start), end),
            end_sec=end,
            confidence=float(row.get("confidence", 1.0)),
            impulsive=bool(self.categories[category].get("impulsive", False)),
            in_gate=category in gate_cats,
            sources=["manual"],
        )

    def detect(self, video_path: Path | None, source_wav: Path, workdir: Path) -> DetectionResult:
        gate_cats = self._gate_categories()
        events = sorted(
            (self._event(row, gate_cats) for row in _event_rows(self.events)),
            key=lambda e: e.start_sec,
        )
        source = str(self.events) if isinstance(self.events, (str, Path)) else "inline"
        return DetectionResult(
            events=events,
            report={
                "detector": {"mode": "manual", "source": source},
                "gate_categories_used": gate_cats,
                "timeline_hz": 100,
                "categories": {
                    name: {"impulsive": bool(cfg.get("impulsive")), "in_gate": name in gate_cats}
                    for name, cfg in self.categories.items()
                },
            },
        )
