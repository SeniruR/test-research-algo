"""Load and query the predefined event taxonomy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.yaml")


@dataclass
class CategoryConfig:
    name: str
    audioset_labels: list[str] = field(default_factory=list)
    kinetics_labels: list[str] = field(default_factory=list)
    context_token_labels: list[str] = field(default_factory=list)
    audio_weight: float = 0.5
    video_weight: float = 0.5
    require_context_or_both: bool = False
    impulsive: bool = False
    include_in_haptic_gate: bool = False


@dataclass
class Taxonomy:
    timeline_hz: int = 100
    context_detector_threshold: float = 0.85
    encoder_threshold: float = 0.35
    impulsive_encoder_threshold: float = 0.30
    sustained_encoder_threshold: float = 0.25
    impulsive_min_peak_distance_sec: float = 0.7
    impulsive_event_half_width_sec: float = 0.45
    impulsive_onset_search_radius_sec: float = 0.75
    impulsive_onset_back_sec: float = 0.6
    impulsive_onset_forward_sec: float = 1.2
    impulsive_short_clip_sec: float = 20.0
    impulsive_decay_tail_sec: float = 0.85
    impulsive_decay_threshold: float = 0.12
    impulsive_pre_roll_sec: float = 0.08
    sustained_max_gate_sec: float = 2.5
    proposal_rms_hop_ms: float = 5.0
    proposal_threshold_ratio: float = 0.25
    proposal_search_pad_sec: float = 0.75
    proposal_window_sec: float = 1.0
    proposal_sustained_hop_sec: float = 2.0
    onset_flux_hop_ms: float = 5.0
    onset_flux_min_ratio: float = 0.45
    categories: dict[str, CategoryConfig] = field(default_factory=dict)

    def all_audioset_labels(self) -> set[str]:
        out: set[str] = set()
        for cat in self.categories.values():
            out.update(cat.audioset_labels)
        return out

    def all_kinetics_labels(self) -> set[str]:
        out: set[str] = set()
        for cat in self.categories.values():
            out.update(cat.kinetics_labels)
        return out


def _normalize(label: str) -> str:
    return " ".join(label.lower().strip().split())


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    path = Path(path) if path else DEFAULT_TAXONOMY_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories: dict[str, CategoryConfig] = {}
    for name, cfg in raw.get("categories", {}).items():
        categories[name] = CategoryConfig(
            name=name,
            audioset_labels=list(cfg.get("audioset_labels", [])),
            kinetics_labels=list(cfg.get("kinetics_labels", [])),
            context_token_labels=list(cfg.get("context_token_labels", [])),
            audio_weight=float(cfg.get("audio_weight", 0.5)),
            video_weight=float(cfg.get("video_weight", 0.5)),
            require_context_or_both=bool(cfg.get("require_context_or_both", False)),
            impulsive=bool(cfg.get("impulsive", False)),
            include_in_haptic_gate=bool(
                cfg.get("include_in_haptic_gate", bool(cfg.get("impulsive", False)))
            ),
        )
    return Taxonomy(
        timeline_hz=int(raw.get("timeline_hz", 100)),
        context_detector_threshold=float(raw.get("context_detector_threshold", 0.85)),
        encoder_threshold=float(raw.get("encoder_threshold", 0.35)),
        impulsive_encoder_threshold=float(raw.get("impulsive_encoder_threshold", 0.30)),
        sustained_encoder_threshold=float(raw.get("sustained_encoder_threshold", 0.25)),
        impulsive_min_peak_distance_sec=float(
            raw.get("impulsive_min_peak_distance_sec", 0.7)
        ),
        impulsive_event_half_width_sec=float(
            raw.get("impulsive_event_half_width_sec", 0.45)
        ),
        impulsive_onset_search_radius_sec=float(
            raw.get("impulsive_onset_search_radius_sec", 0.75)
        ),
        impulsive_onset_back_sec=float(raw.get("impulsive_onset_back_sec", 0.6)),
        impulsive_onset_forward_sec=float(raw.get("impulsive_onset_forward_sec", 1.2)),
        impulsive_short_clip_sec=float(raw.get("impulsive_short_clip_sec", 20.0)),
        impulsive_decay_tail_sec=float(raw.get("impulsive_decay_tail_sec", 0.85)),
        impulsive_decay_threshold=float(raw.get("impulsive_decay_threshold", 0.12)),
        impulsive_pre_roll_sec=float(raw.get("impulsive_pre_roll_sec", 0.08)),
        sustained_max_gate_sec=float(raw.get("sustained_max_gate_sec", 2.5)),
        proposal_rms_hop_ms=float(raw.get("proposal_rms_hop_ms", 5.0)),
        proposal_threshold_ratio=float(raw.get("proposal_threshold_ratio", 0.25)),
        proposal_search_pad_sec=float(raw.get("proposal_search_pad_sec", 0.75)),
        proposal_window_sec=float(raw.get("proposal_window_sec", 1.0)),
        proposal_sustained_hop_sec=float(raw.get("proposal_sustained_hop_sec", 2.0)),
        onset_flux_hop_ms=float(raw.get("onset_flux_hop_ms", 5.0)),
        onset_flux_min_ratio=float(raw.get("onset_flux_min_ratio", 0.45)),
        categories=categories,
    )


def match_label_to_category(taxonomy: Taxonomy, label: str, source: str) -> str | None:
    """Map a raw model label to a taxonomy category name, or None."""
    norm = _normalize(label)
    for cat_name, cat in taxonomy.categories.items():
        if source in ("audio", "context") or source == "audioset":
            pool = cat.audioset_labels + cat.context_token_labels
        elif source in ("video", "kinetics"):
            pool = cat.kinetics_labels
        elif source == "context_token":
            pool = cat.context_token_labels
        else:
            pool = (
                cat.audioset_labels
                + cat.kinetics_labels
                + cat.context_token_labels
            )
        for candidate in pool:
            c_norm = _normalize(candidate)
            if norm == c_norm or c_norm in norm or norm in c_norm:
                return cat_name
    return None
