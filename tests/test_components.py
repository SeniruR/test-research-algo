"""Tests for the plugin architecture: registry, config, isolation, contracts."""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import haptic_gt.components.detectors as detectors_pkg
import haptic_gt.components.generators as generators_pkg
from haptic_gt.core import runner
from haptic_gt.core.config import ComponentSpec, load_pipeline_config, parse_pipeline_config
from haptic_gt.core.registry import (
    ComponentError,
    available_components,
    load_component,
    load_detector,
    load_generators,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS_DIR = REPO_ROOT / "haptic_gt" / "components"
CORE_DIR = REPO_ROOT / "haptic_gt" / "core"


def _write_tone(path: Path, duration: float = 2.0, sr: int = 44100) -> None:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    sf.write(path, 0.5 * np.sin(2 * np.pi * 220 * t), sr, subtype="PCM_16")


def _drop_in(pkg, tmp_path: Path, name: str, source: str):
    """Add a component folder outside the repo, as a user would drop one in."""
    root = tmp_path / pkg.__name__.rsplit(".", 1)[-1]
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    (folder / "component.py").write_text(textwrap.dedent(source), encoding="utf-8")
    pkg.__path__.append(str(root))
    return root


@pytest.fixture
def drop_in(tmp_path):
    added: list[tuple[object, str]] = []

    def _add(pkg, name: str, source: str) -> None:
        root = _drop_in(pkg, tmp_path, name, source)
        added.append((pkg, str(root)))

    yield _add
    for pkg, root in added:
        pkg.__path__.remove(root)
    for mod in [m for m in sys.modules if ".dummy_" in m]:
        del sys.modules[mod]


DUMMY_FULL_MIX = """
    import numpy as np
    import soundfile as sf

    class Component:
        slot = "e"
        output_name = "algorithm_e_dummy.wav"
        input = "full_mix"

        def __init__(self, level=0.25):
            self.level = level

        def generate(self, in_wav, out_wav, ctx):
            info = sf.info(str(in_wav))
            n = int(info.duration * ctx.output_sr)
            sf.write(str(out_wav), np.full(n, self.level, dtype=np.float32), ctx.output_sr)
            return {"level": self.level}
"""

DUMMY_EVENTS = """
    import numpy as np
    import soundfile as sf

    class Component:
        slot = "b"
        output_name = "algorithm_b_dummy.wav"
        input = "events"

        def generate(self, in_wav, out_wav, ctx):
            info = sf.info(str(in_wav))
            n = max(1, int(info.duration * ctx.output_sr))
            sf.write(str(out_wav), np.full(n, 0.5, dtype=np.float32), ctx.output_sr)
"""

DUMMY_DETECTOR = """
    from haptic_gt.core.contracts import DetectionResult, Event

    class Component:
        def detect(self, video_path, source_wav, workdir):
            ev = Event("thump", "Thump", 0.5, 0.6, 0.9, 1.0, impulsive=True, in_gate=True)
            return DetectionResult(events=[ev], report={"detector": {"mode": "dummy"}})
"""


# --- registry -----------------------------------------------------------------


def test_shipped_components_are_discovered():
    assert available_components("detectors") == ["fusion_detector", "manual_events"]
    assert available_components("generators") == [
        "a_perception_mapping",
        "b_frequency_shifting",
        "c_pitch_matching",
        "d_haptic_gen",
        "e_rule_based",
    ]


def test_missing_component_lists_what_is_available():
    with pytest.raises(ComponentError) as err:
        load_component("generators", "z_does_not_exist")
    message = str(err.value)
    assert "z_does_not_exist" in message
    assert "a_perception_mapping" in message


def test_bad_params_name_the_component():
    with pytest.raises(ComponentError, match="b_frequency_shifting"):
        load_component("generators", ComponentSpec("b_frequency_shifting", {"nonsense": 1}))


def test_no_detector_means_none():
    assert load_detector(None) is None
    assert parse_pipeline_config({"detector": None, "generators": []}).detector is None
    assert parse_pipeline_config({"detector": "none", "generators": []}).detector is None


def test_duplicate_slots_are_rejected(drop_in):
    drop_in(generators_pkg, "dummy_slot_clash", DUMMY_FULL_MIX)
    with pytest.raises(ComponentError, match="slot 'e'"):
        load_generators(["e_rule_based", "dummy_slot_clash"])


# --- drop-in components -------------------------------------------------------


def test_dropped_in_generator_runs_without_core_changes(drop_in, tmp_path):
    drop_in(generators_pkg, "dummy_full_mix", DUMMY_FULL_MIX)
    assert "dummy_full_mix" in available_components("generators")

    source = tmp_path / "clip.wav"
    _write_tone(source)
    config = parse_pipeline_config(
        {"detector": None, "generators": [{"name": "dummy_full_mix", "params": {"level": 0.3}}]}
    )
    result = runner.run_pipeline(config, source, tmp_path / "out", from_video=False)

    out = result.outputs["algorithm_e_dummy.wav"]
    audio, sr = sf.read(out)
    assert sr == 8000
    assert audio.size == 2 * 8000
    assert np.allclose(audio, 0.3, atol=1e-3)
    assert result.generator_reports == {"dummy_full_mix": {"level": 0.3}}
    assert result.events_json is None


def test_dropped_in_detector_drives_event_generators(drop_in, tmp_path):
    drop_in(detectors_pkg, "dummy_detector", DUMMY_DETECTOR)
    drop_in(generators_pkg, "dummy_events", DUMMY_EVENTS)

    source = tmp_path / "clip.wav"
    _write_tone(source)
    config = parse_pipeline_config(
        {"detector": "dummy_detector", "generators": ["dummy_events"]}
    )
    result = runner.run_pipeline(config, source, tmp_path / "out", from_video=False)

    audio, sr = sf.read(result.outputs["algorithm_b_dummy.wav"])
    assert audio.size == 2 * sr
    assert np.max(np.abs(audio[: int(0.4 * sr)])) == 0.0
    assert np.max(np.abs(audio[int(0.55 * sr) : int(0.65 * sr)])) > 0.1
    assert result.gated_wav is not None and result.gated_wav.exists()

    payload = json.loads(result.events_json.read_text(encoding="utf-8"))
    assert payload["detector"] == {"mode": "dummy"}
    assert payload["events"][0]["category"] == "thump"
    assert payload["haptic_outputs"] == {
        "gated_audio": "gated_audio.wav",
        "algorithm_b": "algorithm_b_dummy.wav",
    }


# --- config -------------------------------------------------------------------


def test_override_with_new_detector_drops_old_params():
    config = load_pipeline_config(
        overrides={"detector": {"name": "manual_events", "params": {"events": "x.json"}}}
    )
    assert config.detector == ComponentSpec("manual_events", {"events": "x.json"})


def test_override_with_same_detector_merges_params():
    config = load_pipeline_config(
        overrides={"detector": {"name": "fusion_detector", "params": {"use_qwen": False}}}
    )
    assert config.detector.params["use_qwen"] is False
    assert "manual_rumble_peaks" in config.detector.params


def test_override_replaces_generator_list_and_merges_synthesis():
    config = load_pipeline_config(
        overrides={
            "generators": ["e_rule_based"],
            "synthesis": {"continuous": {"enabled": True}},
        }
    )
    assert [g.name for g in config.generators] == ["e_rule_based"]
    assert config.synthesis.continuous.enabled is True
    assert config.synthesis.sustained_mask_min_events == 3


def test_unknown_synthesis_setting_is_rejected():
    with pytest.raises(ValueError, match="bogus"):
        parse_pipeline_config({"synthesis": {"bogus": 1}})


# --- isolation ----------------------------------------------------------------


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            yield node.level, node.module or "", node.lineno


def _component_files():
    for kind_dir in (COMPONENTS_DIR / "detectors", COMPONENTS_DIR / "generators"):
        for comp_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir() and p.name != "__pycache__"):
            for path in comp_dir.rglob("*.py"):
                yield comp_dir, path


def test_components_only_import_core_or_their_own_folder():
    """Removing any one component folder must not break another one."""
    violations = []
    for comp_dir, path in _component_files():
        depth = len(path.relative_to(comp_dir).parts) - 1
        for level, module, lineno in _imports(path):
            where = f"{path.relative_to(REPO_ROOT)}:{lineno}"
            if level == 0:
                if module == "haptic_gt" or (
                    module.startswith("haptic_gt.") and not module.startswith("haptic_gt.core")
                ):
                    violations.append(f"{where} imports {module}")
            elif level > depth + 1:
                violations.append(f"{where} reaches outside its folder ({'.' * level}{module})")
    assert not violations, "\n".join(violations)


def test_core_never_imports_a_component():
    violations = []
    for path in CORE_DIR.rglob("*.py"):
        for level, module, lineno in _imports(path):
            if module.startswith("haptic_gt.components") or level > 1:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_vendored_sound2hap_matches_its_checksums():
    from scripts.verify_sound2hap import verify

    assert verify(upstream=False) == 0


# --- detector contracts -------------------------------------------------------


def test_fusion_events_carry_impulsive_and_gate_flags():
    from haptic_gt.components.detectors.fusion_detector.component import to_event
    from haptic_gt.components.detectors.fusion_detector.frozen_fusion import DetectedEvent
    from haptic_gt.components.detectors.fusion_detector.taxonomy import load_taxonomy

    tax = load_taxonomy()
    blast = DetectedEvent("explosion", "Explosion", 1.0, 1.1, 1.6, 0.8)
    blast.sources = ["audio", "video"]
    rumble = DetectedEvent("vehicle", "Vehicle", 2.0, 3.0, 6.0, 0.5)

    e_blast = to_event(blast, tax, ["explosion"])
    e_rumble = to_event(rumble, tax, ["explosion"])

    assert (e_blast.impulsive, e_blast.in_gate) == (True, True)
    assert (e_rumble.impulsive, e_rumble.in_gate) == (False, False)
    assert e_blast.sources == ["audio", "video"]

    row = e_blast.to_row("ev_0001")
    assert row["category"] == "explosion"
    assert row["impulsive"] is True
    assert row["included_in_gate"] is True
    assert {"context_token", "audio_score", "video_score"} <= set(row)


def test_manual_events_component_uses_its_own_categories(tmp_path):
    from haptic_gt.components.detectors.manual_events.component import Component

    events_file = tmp_path / "events.json"
    events_file.write_text(
        json.dumps(
            {
                "events": [
                    {"category": "gunshot", "peak_sec": 1.0},
                    {"category": "vehicle", "start_sec": 2.0, "peak_sec": 3.0, "end_sec": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = Component(events=str(events_file)).detect(None, tmp_path / "unused.wav", tmp_path)

    gunshot, vehicle = result.events
    assert gunshot.impulsive and gunshot.in_gate
    assert not vehicle.impulsive and vehicle.in_gate
    assert gunshot.start_sec < gunshot.peak_sec < gunshot.end_sec
    assert result.report["detector"]["mode"] == "manual"


def test_pitch_matching_defaults_to_upstream_loudness_fallback():
    gen = load_component("generators", "c_pitch_matching")
    assert gen.mosqito_available is False


def test_iso532_adapter_matches_what_upstream_reads_and_restores_it():
    adapter = pytest.importorskip(
        "haptic_gt.components.generators.c_pitch_matching.mosqito_adapter"
    )
    if not adapter.mosqito_installed():
        pytest.skip("MoSQITo 1.x not installed")
    from types import SimpleNamespace

    noise = np.random.default_rng(0).standard_normal(int(0.3 * 44100)) * 0.1
    results = adapter.loudness_zwtv(noise, 44100, field_type="free")
    n_time = np.asarray(results["N"]).reshape(-1)
    assert results["N_specific"].shape == (n_time.size, 240)

    module = SimpleNamespace(MOSQITO_AVAILABLE=False)
    with adapter.iso532_loudness(module):
        assert module.MOSQITO_AVAILABLE is True
        assert module.loudness_zwtv is adapter.loudness_zwtv
    assert module.MOSQITO_AVAILABLE is False
    assert not hasattr(module, "loudness_zwtv")


def test_fusion_detector_skips_without_video(tmp_path):
    detector = load_detector("fusion_detector")
    result = detector.detect(None, tmp_path / "unused.wav", tmp_path)
    assert result.events == []
    assert result.report["detector"]["mode"] == "skipped"
