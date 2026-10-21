"""Find components by folder name and construct them.

A component is a folder under ``haptic_gt/components/detectors/`` or
``haptic_gt/components/generators/`` containing a ``component.py`` that defines
a class named ``Component``. Its constructor receives the ``params`` given for
it in pipeline.yaml.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from haptic_gt.core.config import ComponentSpec
from haptic_gt.core.contracts import ContextDetector, HapticGenerator

KIND_PACKAGES = {
    "detectors": "haptic_gt.components.detectors",
    "generators": "haptic_gt.components.generators",
}


class ComponentError(LookupError):
    pass


def _package_dirs(kind: str) -> list[Path]:
    if kind not in KIND_PACKAGES:
        raise ValueError(f"Unknown component kind {kind!r}; expected one of {sorted(KIND_PACKAGES)}")
    package = importlib.import_module(KIND_PACKAGES[kind])
    return [Path(p) for p in package.__path__]


def available_components(kind: str) -> list[str]:
    names: set[str] = set()
    for root in _package_dirs(kind):
        for child in root.iterdir():
            if child.is_dir() and (child / "component.py").is_file():
                names.add(child.name)
    return sorted(names)


def load_component(kind: str, spec: ComponentSpec | str) -> Any:
    if isinstance(spec, str):
        spec = ComponentSpec(name=spec)
    names = available_components(kind)
    if spec.name not in names:
        raise ComponentError(
            f"No {kind[:-1]} named {spec.name!r}. "
            f"Available in haptic_gt/components/{kind}/: {', '.join(names) or '(none)'}"
        )
    module = importlib.import_module(f"{KIND_PACKAGES[kind]}.{spec.name}.component")
    cls = getattr(module, "Component", None)
    if cls is None:
        raise ComponentError(f"{module.__name__} does not define a class named Component")
    try:
        instance = cls(**spec.params)
    except TypeError as exc:
        raise ComponentError(f"Bad params for {kind[:-1]} {spec.name!r}: {exc}") from exc
    instance.component_name = spec.name
    return instance


def load_detector(spec: ComponentSpec | str | None) -> ContextDetector | None:
    if spec is None:
        return None
    detector = load_component("detectors", spec)
    if not isinstance(detector, ContextDetector):
        raise ComponentError(f"Detector {detector.component_name!r} has no detect() method")
    return detector


def load_generators(specs: list[ComponentSpec | str]) -> list[HapticGenerator]:
    generators: list[HapticGenerator] = []
    seen_slots: dict[str, str] = {}
    seen_names: dict[str, str] = {}
    for spec in specs:
        gen = load_component("generators", spec)
        name = gen.component_name
        if not isinstance(gen, HapticGenerator):
            raise ComponentError(
                f"Generator {name!r} must define slot, output_name, input and generate()"
            )
        if gen.input not in ("events", "full_mix"):
            raise ComponentError(f"Generator {name!r}: input must be 'events' or 'full_mix'")
        if not str(gen.output_name).lower().endswith(".wav"):
            raise ComponentError(f"Generator {name!r}: output_name must end in .wav")
        if gen.slot in seen_slots:
            raise ComponentError(
                f"Generators {seen_slots[gen.slot]!r} and {name!r} both use slot {gen.slot!r}"
            )
        if gen.output_name in seen_names:
            raise ComponentError(
                f"Generators {seen_names[gen.output_name]!r} and {name!r} both write {gen.output_name!r}"
            )
        seen_slots[gen.slot] = name
        seen_names[gen.output_name] = name
        generators.append(gen)
    return generators
