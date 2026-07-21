"""End-to-end pipeline aligned with Sound2Hap signal processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from haptic_gt.algorithms import freq_shift, haptic_gen, percept, pitch_match
from haptic_gt.audio_io import INPUT_SR, VIB_SR, extract_audio_from_video, prepare_source_wav

OUTPUT_NAMES = {
    "source_audio": "source_audio.wav",
    "algorithm_a_perception_mapping": "algorithm_a_perception_mapping.wav",
    "algorithm_b_frequency_shifting": "algorithm_b_frequency_shifting.wav",
    "algorithm_c_pitch_matching": "algorithm_c_pitch_matching.wav",
    "algorithm_d_haptic_gen": "algorithm_d_haptic_gen.wav",
}


@dataclass
class CandidateTracks:
    """Paths to source audio and four Sound2Hap candidate haptic tracks."""

    source_wav: Path
    algorithm_a: Path
    algorithm_b: Path
    algorithm_c: Path
    algorithm_d: Path
    output_dir: Path
    input_sample_rate: int = INPUT_SR
    output_sample_rate: int = VIB_SR
    pitch_match_info: dict | None = None

    def save_all(self) -> dict[str, Path]:
        return {
            "source_audio": self.source_wav,
            "algorithm_a_perception_mapping": self.algorithm_a,
            "algorithm_b_frequency_shifting": self.algorithm_b,
            "algorithm_c_pitch_matching": self.algorithm_c,
            "algorithm_d_haptic_gen": self.algorithm_d,
        }


def generate_candidate_tracks(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    from_video: bool = True,
    content_type: str = "game",
) -> CandidateTracks:
    """
    Run the Sound2Hap processing engine on one video or audio file.

    Parameters
    ----------
    input_path:
        Video (mp4, mov, ...) or audio (wav, mp3, ...) path.
    output_dir:
        Directory for 44.1 kHz source + 8 kHz haptic WAV outputs.
    from_video:
        Extract audio with ffmpeg when True.
    content_type:
        Perceptual mapping content profile: ``"game"`` (games/movies) or ``"music"``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_wav = output_dir / OUTPUT_NAMES["source_audio"]
    if from_video:
        extract_audio_from_video(input_path, source_wav, sr=INPUT_SR)
    else:
        prepare_source_wav(input_path, source_wav, sr=INPUT_SR)

    out_a = output_dir / OUTPUT_NAMES["algorithm_a_perception_mapping"]
    out_b = output_dir / OUTPUT_NAMES["algorithm_b_frequency_shifting"]
    out_c = output_dir / OUTPUT_NAMES["algorithm_c_pitch_matching"]
    out_d = output_dir / OUTPUT_NAMES["algorithm_d_haptic_gen"]

    percept.process_file(source_wav, out_a, content=content_type)
    freq_shift.process_file(source_wav, out_b)
    pitch_info = pitch_match.process_file(source_wav, out_c)
    haptic_gen.process_file(source_wav, out_d)

    return CandidateTracks(
        source_wav=source_wav,
        algorithm_a=out_a,
        algorithm_b=out_b,
        algorithm_c=out_c,
        algorithm_d=out_d,
        output_dir=output_dir,
        pitch_match_info=pitch_info,
    )
