"""End-to-end pipeline: video/audio in -> four candidate haptic tracks out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithm_a import perception_mapping
from .algorithm_b import frequency_shifting
from .algorithm_c import pitch_matching
from .algorithm_d import haptic_gen
from .audio_io import TARGET_SR, extract_audio_from_video, load_audio, save_haptic


@dataclass
class CandidateTracks:
    """Paths and in-memory arrays for all four algorithm outputs."""

    source_audio: np.ndarray
    sample_rate: int
    algorithm_a: np.ndarray
    algorithm_b: np.ndarray
    algorithm_c: np.ndarray
    algorithm_d: np.ndarray
    output_dir: Path

    def save_all(self) -> dict[str, Path]:
        """Write source audio and all candidate tracks to output_dir."""
        sr = self.sample_rate
        out = {
            "source_audio": save_haptic(
                self.output_dir / "source_audio.wav", self.source_audio, sr
            ),
            "algorithm_a_perception_mapping": save_haptic(
                self.output_dir / "algorithm_a_perception_mapping.wav",
                self.algorithm_a,
                sr,
            ),
            "algorithm_b_frequency_shifting": save_haptic(
                self.output_dir / "algorithm_b_frequency_shifting.wav",
                self.algorithm_b,
                sr,
            ),
            "algorithm_c_pitch_matching": save_haptic(
                self.output_dir / "algorithm_c_pitch_matching.wav",
                self.algorithm_c,
                sr,
            ),
            "algorithm_d_haptic_gen": save_haptic(
                self.output_dir / "algorithm_d_haptic_gen.wav",
                self.algorithm_d,
                sr,
            ),
        }
        return out


def generate_candidate_tracks(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    from_video: bool = True,
    target_rms: float = 0.1,
    sr: int = TARGET_SR,
) -> CandidateTracks:
    """
    Run the full processing engine on one video or audio file.

    Parameters
    ----------
    input_path:
        Path to a video (mp4, mov, ...) or audio file (wav, mp3, ...).
    output_dir:
        Directory where WAV outputs will be written.
    from_video:
        If True, treat input as video and extract audio via ffmpeg.
        If False, load input directly as audio.
    target_rms:
        Target RMS level for all candidate tracks after normalization.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if from_video:
        audio, loaded_sr = extract_audio_from_video(
            input_path,
            output_path=output_dir / "source_audio.wav",
            sr=sr,
        )
    else:
        audio, loaded_sr = load_audio(input_path, sr=sr)

    return CandidateTracks(
        source_audio=audio,
        sample_rate=loaded_sr,
        algorithm_a=perception_mapping(audio, loaded_sr, target_rms),
        algorithm_b=frequency_shifting(audio, loaded_sr, target_rms),
        algorithm_c=pitch_matching(audio, loaded_sr, target_rms),
        algorithm_d=haptic_gen(audio, loaded_sr, target_rms),
        output_dir=output_dir,
    )
