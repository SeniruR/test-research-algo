"""A: perception-level mapping (Lee & Choi, CHI 2013).

Runs the unmodified ``Percept.py`` from third_party/sound2hap.
"""

from __future__ import annotations

from pathlib import Path

from haptic_gt.core.contracts import GeneratorContext
from haptic_gt.core.vendor import THIRD_PARTY_DIR, import_from_dir, quiet

SOUND2HAP_DIR = THIRD_PARTY_DIR / "sound2hap"


class Component:
    slot = "a"
    output_name = "algorithm_a_perception_mapping.wav"
    input = "events"

    def __init__(self, content: str = "game"):
        """``content``: "game" for games/movies, "music" for music (upstream option)."""
        self.content = content
        self._percept = import_from_dir(SOUND2HAP_DIR, "Percept")

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with quiet():
            self._percept.process_file(str(in_wav), str(out_wav), content=self.content)
