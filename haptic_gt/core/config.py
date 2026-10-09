"""Load pipeline.yaml: which components run, with what parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from haptic_gt.core.synthesis import (
    SUSTAINED_MASK_MIN_COVERAGE,
    SUSTAINED_MASK_MIN_EVENTS,
    ContinuousProfile,
)
from haptic_gt.core.vendor import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "pipeline.yaml"


@dataclass
class ComponentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisConfig:
    continuous: ContinuousProfile = field(
        default_factory=lambda: ContinuousProfile(enabled=False)
    )
    sustained_mask_min_events: int = SUSTAINED_MASK_MIN_EVENTS
    sustained_mask_min_coverage: float = SUSTAINED_MASK_MIN_COVERAGE


@dataclass
class PipelineConfig:
    detector: ComponentSpec | None
    generators: list[ComponentSpec]
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _merge(current, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _component_spec(raw: Any, where: str) -> ComponentSpec:
    if isinstance(raw, str):
        return ComponentSpec(name=raw)
    if isinstance(raw, dict) and raw.get("name"):
        return ComponentSpec(name=str(raw["name"]), params=dict(raw.get("params") or {}))
    raise ValueError(f"{where}: expected a component name or {{name, params}}, got {raw!r}")


def _continuous_profile(raw: dict[str, Any]) -> ContinuousProfile:
    known = {f.name for f in fields(ContinuousProfile)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"synthesis.continuous: unknown setting(s) {unknown}; expected some of {sorted(known)}"
        )
    return ContinuousProfile(**{"enabled": False, **raw})


def parse_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    detector_raw = raw.get("detector")
    detector = None if detector_raw in (None, "", "none") else _component_spec(detector_raw, "detector")

    generators = [
        _component_spec(item, f"generators[{i}]")
        for i, item in enumerate(raw.get("generators") or [])
    ]

    synth_raw = dict(raw.get("synthesis") or {})
    synthesis = SynthesisConfig(
        continuous=_continuous_profile(dict(synth_raw.pop("continuous", None) or {})),
        sustained_mask_min_events=int(
            synth_raw.pop("sustained_mask_min_events", SUSTAINED_MASK_MIN_EVENTS)
        ),
        sustained_mask_min_coverage=float(
            synth_raw.pop("sustained_mask_min_coverage", SUSTAINED_MASK_MIN_COVERAGE)
        ),
    )
    if synth_raw:
        raise ValueError(f"synthesis: unknown setting(s) {sorted(synth_raw)}")
    return PipelineConfig(detector=detector, generators=generators, synthesis=synthesis)


def load_pipeline_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Read pipeline.yaml and apply ``overrides`` on top.

    Overrides merge into nested settings, except that a ``detector`` with a
    different name replaces the configured one outright (its params belong to
    the old component), and ``generators`` is always replaced as a whole list.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if overrides:
        overrides = dict(overrides)
        detector_override = overrides.get("detector")
        base_detector = raw.get("detector")
        base_name = base_detector.get("name") if isinstance(base_detector, dict) else base_detector
        if isinstance(detector_override, dict) and detector_override.get("name") not in (None, base_name):
            raw["detector"] = overrides.pop("detector")
        elif isinstance(detector_override, str) and detector_override != base_name:
            raw["detector"] = overrides.pop("detector")
        raw = _merge(raw, overrides)

    return parse_pipeline_config(raw)
