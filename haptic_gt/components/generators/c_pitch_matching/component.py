"""C: sound-to-touch pitch matching (Kim et al., IEEE ToH 2023).

Runs the unmodified ``Pitch_WebTool.py`` from third_party/sound2hap.

Upstream's MoSQITo import never succeeds (see mosqito_adapter.py), so by default
the specific loudness is upstream's mean-envelope fallback, exactly as the
published script behaves. ``iso532_loudness: true`` feeds it real ISO 532-1
loudness from MoSQITo 1.x instead.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from haptic_gt.core.contracts import GeneratorContext
from haptic_gt.core.vendor import THIRD_PARTY_DIR, import_from_dir, quiet

from . import mosqito_adapter

SOUND2HAP_DIR = THIRD_PARTY_DIR / "sound2hap"


class Component:
    slot = "c"
    output_name = "algorithm_c_pitch_matching.wav"
    input = "events"

    def __init__(self, iso532_loudness: bool = False):
        if iso532_loudness and not mosqito_adapter.mosqito_installed():
            raise RuntimeError("iso532_loudness needs MoSQITo 1.x: pip install 'mosqito>=1.2,<2'")
        self.iso532_loudness = iso532_loudness
        self._pitch = import_from_dir(SOUND2HAP_DIR, "Pitch_WebTool")
        self._cfg = self._pitch.get_config()

    @property
    def mosqito_available(self) -> bool:
        """False means the specific loudness is upstream's mean-envelope fallback."""
        return self.iso532_loudness or bool(self._pitch.MOSQITO_AVAILABLE)

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext) -> dict[str, Any]:
        loudness = (
            mosqito_adapter.iso532_loudness(self._pitch)
            if self.iso532_loudness
            else contextlib.nullcontext()
        )
        with loudness, quiet():
            _ok, info = self._pitch.process_audio_file(str(in_wav), str(out_wav), self._cfg)
        return {
            "mosqito_available": self.mosqito_available,
            "loudness": "iso532_1" if self.mosqito_available else "envelope_fallback",
            "analysis_info": info.get("analysisInfo"),
        }
