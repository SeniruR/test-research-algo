"""D: HapticGen-style RMS-driven oscillator (Sung et al., CHI 2025).

Runs the unmodified ``HapticGen.py`` from third_party/sound2hap.
"""

from __future__ import annotations

from pathlib import Path

from haptic_gt.core.contracts import GeneratorContext
from haptic_gt.core.vendor import THIRD_PARTY_DIR, import_from_dir, quiet

SOUND2HAP_DIR = THIRD_PARTY_DIR / "sound2hap"


class Component:
    slot = "d"
    output_name = "algorithm_d_haptic_gen.wav"
    input = "events"

    def __init__(self):
        self._haptic_gen = import_from_dir(SOUND2HAP_DIR, "HapticGen")

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with quiet():
            self._haptic_gen.process_file(str(in_wav), str(out_wav))
