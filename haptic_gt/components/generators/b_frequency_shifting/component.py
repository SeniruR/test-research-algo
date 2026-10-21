"""B: frequency shifting (Okazaki et al., 2015).

Runs the unmodified ``FreqShift.py`` from third_party/sound2hap.
"""

from __future__ import annotations

from pathlib import Path

from haptic_gt.core.contracts import GeneratorContext
from haptic_gt.core.vendor import THIRD_PARTY_DIR, import_from_dir, quiet

SOUND2HAP_DIR = THIRD_PARTY_DIR / "sound2hap"


class Component:
    slot = "b"
    output_name = "algorithm_b_frequency_shifting.wav"
    input = "events"

    def __init__(self, centre_hz: float = 250.0, q: float = 1.0):
        self.centre_hz = centre_hz
        self.q = q
        self._freq_shift = import_from_dir(SOUND2HAP_DIR, "FreqShift")

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext) -> None:
        with quiet():
            self._freq_shift.process_file(
                str(in_wav), str(out_wav), centre_hz=self.centre_hz, q=self.q
            )
