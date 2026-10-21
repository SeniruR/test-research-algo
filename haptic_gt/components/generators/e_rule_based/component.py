"""E: the original rule-based thunder / rain / audio-RMS mapper.

Runs once on the ungated mix, plus video frames when the input is a video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haptic_gt.core.contracts import GeneratorContext

from . import rule_based


class Component:
    slot = "e"
    output_name = "algorithm_e_rule_based.wav"
    input = "full_mix"

    def __init__(self, use_video: bool = True, write_json: bool = True):
        self.use_video = use_video
        self.write_json = write_json

    def generate(self, in_wav: Path, out_wav: Path, ctx: GeneratorContext) -> dict[str, Any]:
        json_path = ctx.debug_dir / "algorithm_e_rule_based.json" if self.write_json else None
        payload = rule_based.process_file(
            in_wav,
            out_wav,
            video_path=ctx.video_path if self.use_video else None,
            json_path=json_path,
        )
        return {
            "windows": len(payload.get("events", [])),
            "used_video": bool(self.use_video and ctx.video_path is not None),
            "json_path": payload.get("json_path"),
        }
