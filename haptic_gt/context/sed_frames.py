"""Frame-level sound event posteriors (DCASE-style), not clip-level tagging.

Window tagging gives one score per ~1 s window, so onsets can only be located
to about a second — that is why timing came from spectral flux alone before.
Here every category gets a posterior on a fixed frame grid, so onsets come from
the model's own time axis.

Backends
--------
``ast_dense``
    AST (AudioSet) evaluated on densely overlapping windows (hop ``sed_hop_sec``),
    using all 527 sigmoid outputs instead of a truncated top-k. Resolution is the
    hop; effective smoothing is the window length. No extra download.
``panns``
    PANNs ``Cnn14_DecisionLevelAtt`` framewise output (true frame-level SED, the
    DCASE-style option). Used when ``panns_inference`` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from haptic_gt.context.encoders import EncoderScore
from haptic_gt.context.taxonomy import Taxonomy, load_taxonomy, match_label_to_category

AST_SR = 16_000
PANNS_SR = 32_000

_ast_model: Any | None = None
_ast_extractor: Any | None = None
_panns_model: Any | None = None


@dataclass
class FramePosteriors:
    """Per-category posterior on a uniform frame grid."""

    times: np.ndarray  # [T] frame center times (s)
    scores: dict[str, np.ndarray]  # category -> [T] posterior
    hop_sec: float
    backend: str
    duration_sec: float

    def category_names(self) -> list[str]:
        return sorted(self.scores)

    def at(self, category: str, t: float) -> float:
        arr = self.scores.get(category)
        if arr is None or arr.size == 0 or self.times.size == 0:
            return 0.0
        return float(arr[int(np.argmin(np.abs(self.times - t)))])


def _device_str() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_ast():
    """Load AST model + feature extractor directly (all logits, batched)."""
    global _ast_model, _ast_extractor
    if _ast_model is None or _ast_extractor is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        from haptic_gt.context.encoders import AST_MODEL_ID

        _ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
        _ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID)
        _ast_model.eval()
        _ast_model.to(_device_str())
        for p in _ast_model.parameters():
            p.requires_grad = False
        torch.set_grad_enabled(False)
    return _ast_model, _ast_extractor


def _category_label_index(taxonomy: Taxonomy, id2label: dict) -> dict[str, list[int]]:
    """Map each taxonomy category to the model output indices that feed it."""
    out: dict[str, list[int]] = {name: [] for name in taxonomy.categories}
    for idx, label in id2label.items():
        cat = match_label_to_category(taxonomy, str(label), "audio")
        if cat is not None:
            out[cat].append(int(idx))
    return {k: v for k, v in out.items() if v}


def _ast_dense_posteriors(
    audio_16k: np.ndarray,
    taxonomy: Taxonomy,
    *,
    batch_size: int = 12,
) -> FramePosteriors:
    import torch

    model, extractor = _get_ast()
    device = _device_str()
    hop = taxonomy.sed_hop_sec
    win = taxonomy.sed_window_sec
    duration = len(audio_16k) / float(AST_SR)

    centers: list[float] = []
    clips: list[np.ndarray] = []
    t = 0.0
    while t <= max(duration - 1e-6, 0.0):
        s0 = int(max(0.0, t - win / 2.0) * AST_SR)
        s1 = int(min(duration, t + win / 2.0) * AST_SR)
        clip = audio_16k[s0:s1]
        if clip.size >= AST_SR // 20:
            centers.append(t)
            clips.append(clip.astype(np.float32))
        t += hop

    cat_idx = _category_label_index(taxonomy, model.config.id2label)
    scores: dict[str, list[float]] = {c: [] for c in cat_idx}

    for i in range(0, len(clips), batch_size):
        batch = clips[i : i + batch_size]
        inputs = extractor(
            batch, sampling_rate=AST_SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        # AudioSet head is multi-label: sigmoid, and keep every class
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        for row in probs:
            for cat, idxs in cat_idx.items():
                scores[cat].append(float(np.max(row[idxs])))

    return FramePosteriors(
        times=np.asarray(centers, dtype=np.float64),
        scores={c: np.asarray(v, dtype=np.float64) for c, v in scores.items()},
        hop_sec=hop,
        backend="ast_dense",
        duration_sec=duration,
    )


def _panns_posteriors(audio_16k: np.ndarray, taxonomy: Taxonomy) -> FramePosteriors:
    """True framewise SED via PANNs Cnn14_DecisionLevelAtt (~10 ms frames)."""
    global _panns_model
    import librosa
    from panns_inference import SoundEventDetection, labels

    if _panns_model is None:
        _panns_model = SoundEventDetection(checkpoint_path=None, device=_device_str())

    audio_32k = librosa.resample(
        audio_16k.astype(np.float32), orig_sr=AST_SR, target_sr=PANNS_SR
    )
    framewise = _panns_model.inference(audio_32k[None, :])[0]  # [T, 527]
    duration = len(audio_16k) / float(AST_SR)
    n_frames = framewise.shape[0]
    hop = duration / max(n_frames, 1)
    times = (np.arange(n_frames) + 0.5) * hop

    id2label = {i: lab for i, lab in enumerate(labels)}
    cat_idx = _category_label_index(taxonomy, id2label)
    scores = {
        cat: np.max(framewise[:, idxs], axis=1).astype(np.float64)
        for cat, idxs in cat_idx.items()
    }
    return FramePosteriors(
        times=times,
        scores=scores,
        hop_sec=hop,
        backend="panns",
        duration_sec=duration,
    )


def compute_frame_posteriors(
    audio_16k: np.ndarray,
    taxonomy: Taxonomy | None = None,
    *,
    backend: str | None = None,
) -> FramePosteriors:
    """Frame-level posteriors per taxonomy category."""
    taxonomy = taxonomy or load_taxonomy()
    backend = backend or taxonomy.sed_backend
    if backend in ("auto", "panns"):
        try:
            return _panns_posteriors(audio_16k, taxonomy)
        except Exception:
            if backend == "panns":
                raise
    return _ast_dense_posteriors(audio_16k, taxonomy)


def posteriors_to_encoder_scores(
    frames: FramePosteriors,
    taxonomy: Taxonomy | None = None,
) -> list[EncoderScore]:
    """Frame posteriors as sparse scores, for stages that expect AST hits.

    Each category is represented by its first taxonomy label so existing
    label-to-category matching keeps working.
    """
    taxonomy = taxonomy or load_taxonomy()
    out: list[EncoderScore] = []
    for cat, arr in frames.scores.items():
        cfg = taxonomy.categories.get(cat)
        label = cfg.audioset_labels[0] if cfg and cfg.audioset_labels else cat
        for t, v in zip(frames.times, arr):
            if v <= 0.01:
                continue
            out.append(
                EncoderScore(
                    time_sec=float(t), label=label, score=float(v), source="audio"
                )
            )
    out.sort(key=lambda s: s.time_sec)
    return out
