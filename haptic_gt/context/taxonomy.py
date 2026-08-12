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
    # Optional per-category SED hysteresis overrides
    sed_high: float | None = None
    sed_low: float | None = None


@dataclass
class Taxonomy:
    timeline_hz: int = 100
    context_detector_threshold: float = 0.85
    encoder_threshold: float = 0.35
    impulsive_encoder_threshold: float = 0.30
    sustained_encoder_threshold: float = 0.25
    impulsive_min_peak_distance_sec: float = 0.45
    impulsive_event_half_width_sec: float = 0.45
    impulsive_onset_search_radius_sec: float = 0.75
    impulsive_onset_back_sec: float = 0.35
    impulsive_onset_forward_sec: float = 1.2
    impulsive_short_clip_sec: float = 8.0
    impulsive_decay_tail_sec: float = 0.85
    impulsive_decay_threshold: float = 0.12
    impulsive_pre_roll_sec: float = 0.08
    sustained_max_gate_sec: float = 2.5
    sustained_merge_gap_sec: float = 0.35
    sustained_burst_rise_ratio: float = 1.10
    sustained_burst_pre_sec: float = 0.7
    sustained_burst_post_sec: float = 1.5
    sustained_burst_max_sec: float = 2.5
    sustained_salience_mid_pct: float = 50.0
    sustained_salience_loud_pct: float = 99.0
    sustained_salience_mix: float = 0.40
    sustained_salience_min_sep: float = 1.45
    sustained_salience_silence_frac: float = 0.05
    sustained_rhythm_min_gaps: int = 2
    sustained_rhythm_gap_max_sec: float = 2.0
    sustained_rhythm_level_ratio: float = 2.0
    sustained_scene_level_ratio: float = 2.5
    sustained_scene_min_sec: float = 3.5
    sustained_scene_block_sec: float = 2.0
    sustained_scene_stable_ratio: float = 2.0
    sustained_salience_scene_floor_frac: float = 0.16
    sustained_island_edge_frac: float = 0.75
    sustained_island_max_extend_sec: float = 0.4
    sustained_salience_min_sec: float = 0.50
    sustained_salience_gap_sec: float = 0.18
    sustained_salience_peak_win_sec: float = 0.15
    sustained_salience_min_duty: float = 0.55
    sustained_salience_max_orphan_sec: float = 1.2
    proposal_rms_hop_ms: float = 5.0
    proposal_threshold_ratio: float = 0.25
    proposal_search_pad_sec: float = 0.75
    proposal_window_sec: float = 1.0
    proposal_sustained_hop_sec: float = 1.0
    onset_flux_hop_ms: float = 5.0
    onset_flux_min_ratio: float = 0.45
    onset_flux_early_rel: float = 0.55
    onset_flux_proposal_ratio: float = 0.16
    # Intermittent rumble: mask continuous bed by sustained spans
    sustained_mask_min_events: int = 3
    sustained_mask_min_coverage: float = 0.15
    # Frame-level SED decoding
    sed_enabled: bool = True
    sed_backend: str = "auto"
    sed_window_sec: float = 1.0
    sed_hop_sec: float = 0.1
    sed_median_impulsive_sec: float = 0.15
    sed_median_sustained_sec: float = 0.45
    sed_onset_high: float = 0.30
    sed_onset_low: float = 0.15
    sed_min_event_sec_impulsive: float = 0.10
    sed_min_event_sec_sustained: float = 0.40
    sed_merge_gap_sec: float = 0.20
    sed_peak_rel: float = 0.60
    # Video fusion is off; see taxonomy.yaml for why
    use_video: bool = False
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
            sed_high=(
                float(cfg["sed_high"]) if cfg.get("sed_high") is not None else None
            ),
            sed_low=(float(cfg["sed_low"]) if cfg.get("sed_low") is not None else None),
        )
    return Taxonomy(
        timeline_hz=int(raw.get("timeline_hz", 100)),
        context_detector_threshold=float(raw.get("context_detector_threshold", 0.85)),
        encoder_threshold=float(raw.get("encoder_threshold", 0.35)),
        impulsive_encoder_threshold=float(raw.get("impulsive_encoder_threshold", 0.30)),
        sustained_encoder_threshold=float(raw.get("sustained_encoder_threshold", 0.28)),
        impulsive_min_peak_distance_sec=float(
            raw.get("impulsive_min_peak_distance_sec", 0.45)
        ),
        impulsive_event_half_width_sec=float(
            raw.get("impulsive_event_half_width_sec", 0.45)
        ),
        impulsive_onset_search_radius_sec=float(
            raw.get("impulsive_onset_search_radius_sec", 0.75)
        ),
        impulsive_onset_back_sec=float(raw.get("impulsive_onset_back_sec", 0.35)),
        impulsive_onset_forward_sec=float(raw.get("impulsive_onset_forward_sec", 1.2)),
        impulsive_short_clip_sec=float(raw.get("impulsive_short_clip_sec", 8.0)),
        impulsive_decay_tail_sec=float(raw.get("impulsive_decay_tail_sec", 0.85)),
        impulsive_decay_threshold=float(raw.get("impulsive_decay_threshold", 0.12)),
        impulsive_pre_roll_sec=float(raw.get("impulsive_pre_roll_sec", 0.08)),
        sustained_max_gate_sec=float(raw.get("sustained_max_gate_sec", 2.5)),
        sustained_merge_gap_sec=float(raw.get("sustained_merge_gap_sec", 0.35)),
        sustained_burst_rise_ratio=float(raw.get("sustained_burst_rise_ratio", 1.10)),
        sustained_burst_pre_sec=float(raw.get("sustained_burst_pre_sec", 0.7)),
        sustained_burst_post_sec=float(raw.get("sustained_burst_post_sec", 1.5)),
        sustained_burst_max_sec=float(raw.get("sustained_burst_max_sec", 2.5)),
        sustained_salience_mid_pct=float(raw.get("sustained_salience_mid_pct", 50.0)),
        sustained_salience_loud_pct=float(raw.get("sustained_salience_loud_pct", 99.0)),
        sustained_salience_mix=float(raw.get("sustained_salience_mix", 0.40)),
        sustained_salience_min_sep=float(raw.get("sustained_salience_min_sep", 1.45)),
        sustained_salience_silence_frac=float(
            raw.get("sustained_salience_silence_frac", 0.05)
        ),
        sustained_rhythm_min_gaps=int(raw.get("sustained_rhythm_min_gaps", 2)),
        sustained_rhythm_gap_max_sec=float(
            raw.get("sustained_rhythm_gap_max_sec", 2.0)
        ),
        sustained_rhythm_level_ratio=float(
            raw.get("sustained_rhythm_level_ratio", 2.0)
        ),
        sustained_scene_level_ratio=float(raw.get("sustained_scene_level_ratio", 2.5)),
        sustained_scene_min_sec=float(raw.get("sustained_scene_min_sec", 3.5)),
        sustained_scene_block_sec=float(raw.get("sustained_scene_block_sec", 2.0)),
        sustained_scene_stable_ratio=float(
            raw.get("sustained_scene_stable_ratio", 2.0)
        ),
        sustained_salience_scene_floor_frac=float(
            raw.get("sustained_salience_scene_floor_frac", 0.16)
        ),
        sustained_island_edge_frac=float(raw.get("sustained_island_edge_frac", 0.75)),
        sustained_island_max_extend_sec=float(
            raw.get("sustained_island_max_extend_sec", 0.4)
        ),
        sustained_salience_min_sec=float(raw.get("sustained_salience_min_sec", 0.50)),
        sustained_salience_gap_sec=float(raw.get("sustained_salience_gap_sec", 0.18)),
        sustained_salience_peak_win_sec=float(raw.get("sustained_salience_peak_win_sec", 0.15)),
        sustained_salience_min_duty=float(raw.get("sustained_salience_min_duty", 0.55)),
        sustained_salience_max_orphan_sec=float(
            raw.get("sustained_salience_max_orphan_sec", 1.2)
        ),
        proposal_rms_hop_ms=float(raw.get("proposal_rms_hop_ms", 5.0)),
        proposal_threshold_ratio=float(raw.get("proposal_threshold_ratio", 0.25)),
        proposal_search_pad_sec=float(raw.get("proposal_search_pad_sec", 0.75)),
        proposal_window_sec=float(raw.get("proposal_window_sec", 1.0)),
        proposal_sustained_hop_sec=float(raw.get("proposal_sustained_hop_sec", 1.0)),
        onset_flux_hop_ms=float(raw.get("onset_flux_hop_ms", 5.0)),
        onset_flux_min_ratio=float(raw.get("onset_flux_min_ratio", 0.45)),
        onset_flux_early_rel=float(raw.get("onset_flux_early_rel", 0.55)),
        onset_flux_proposal_ratio=float(raw.get("onset_flux_proposal_ratio", 0.16)),
        sustained_mask_min_events=int(raw.get("sustained_mask_min_events", 3)),
        sustained_mask_min_coverage=float(raw.get("sustained_mask_min_coverage", 0.15)),
        sed_enabled=bool(raw.get("sed_enabled", True)),
        sed_backend=str(raw.get("sed_backend", "auto")),
        sed_window_sec=float(raw.get("sed_window_sec", 1.0)),
        sed_hop_sec=float(raw.get("sed_hop_sec", 0.1)),
        sed_median_impulsive_sec=float(raw.get("sed_median_impulsive_sec", 0.15)),
        sed_median_sustained_sec=float(raw.get("sed_median_sustained_sec", 0.45)),
        sed_onset_high=float(raw.get("sed_onset_high", 0.30)),
        sed_onset_low=float(raw.get("sed_onset_low", 0.15)),
        sed_min_event_sec_impulsive=float(raw.get("sed_min_event_sec_impulsive", 0.10)),
        sed_min_event_sec_sustained=float(raw.get("sed_min_event_sec_sustained", 0.40)),
        sed_merge_gap_sec=float(raw.get("sed_merge_gap_sec", 0.20)),
        sed_peak_rel=float(raw.get("sed_peak_rel", 0.60)),
        use_video=bool(raw.get("use_video", False)),
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
