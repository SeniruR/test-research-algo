"""Scene labels from the picture (Qwen2.5-VL), for the video branch.

The video is cut at scene changes and each cut is asked which haptic categories
are visible. A scene is seconds long and the model sees a few frames per second,
so these spans are context, never bang times: a muzzle flash is one or two
frames and usually falls between samples. Timing comes from the audio branch
and the orange-flash scan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import Taxonomy, load_taxonomy

SCENE_CATEGORIES = ("gunshot", "explosion", "car_crash", "smash", "vehicle", "engine_start")
BANG_SCENE_CATEGORIES = ("gunshot", "explosion", "car_crash", "smash")

PROMPT = (
    "Classify this video segment for haptic context.\n"
    "List every category that is visibly present. A segment may have more than one.\n"
    "- gunshot: a person firing a gun, visible gunfire\n"
    "- explosion: a blast, fireball, artillery impact\n"
    "- car_crash: a vehicle colliding with another vehicle, a wall, or an object\n"
    "- smash: something breaking or being smashed, glass shattering, an object hit hard\n"
    "- vehicle: a car, truck, or tank driving or idling\n"
    "- engine_start: a vehicle engine being started, ignition, first rev\n"
    "car_crash may be listed together with smash; engine_start together with vehicle.\n"
    "Reply with JSON only, no other text. Example for a car hitting a wall: "
    '{"categories": ["car_crash", "smash"]}\n'
    "Use an empty list if none of these are visible: "
    '{"categories": []}'
)


@dataclass
class VisualSpan:
    category: str
    start_sec: float
    end_sec: float


def parse_categories(text: str) -> list[str]:
    """Category names from the model's JSON reply; anything else is ignored."""
    names: list = []
    for match in re.finditer(r"\{.*?\}|\[.*?\]", text, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            names.extend(payload)
        elif isinstance(payload, dict):
            raw = payload.get("categories", payload.get("category", []))
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                names.extend(raw)
    if not names:
        names = [part.strip() for part in text.strip().lower().split(",")]
    found: list[str] = []
    for name in names:
        name = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        if name in SCENE_CATEGORIES and name not in found:
            found.append(name)
    return found


def sample_fps(start: float, end: float, *, max_frames: int, min_look_sec: float) -> float:
    """Denser frames on a short cut, never more than ``max_frames``."""
    duration = max(end - start, min_look_sec)
    if duration <= 4.0:
        target = 8.0
    elif duration <= 8.0:
        target = 4.0
    else:
        target = 2.0
    return round(min(target, max_frames / duration), 2)


def look_window(
    start: float, end: float, duration: float, *, min_look_sec: float
) -> tuple[float, float]:
    """Widen a tiny cut so the model receives at least ``min_look_sec`` of frames."""
    if end - start >= min_look_sec:
        return start, end
    look_start = max(0.0, start - (min_look_sec - (end - start)) / 2.0)
    look_end = look_start + min_look_sec
    if duration and look_end > duration:
        look_end = duration
        look_start = max(0.0, look_end - min_look_sec)
    return look_start, look_end


def scene_chunks(video_path: str | Path, threshold: float = 30.0) -> tuple[list[tuple[float, float]], float]:
    """Scene cuts as (start, end) seconds, plus the video length. Short cuts are kept."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    duration = float(getattr(video.duration, "seconds", 0.0) or 0.0)
    if scenes:
        chunks = [(float(s[0].seconds), float(s[1].seconds)) for s in scenes]
    else:
        chunks = [(0.0, duration)]
    return [(a, b) for a, b in chunks if b > a], duration


def classify_scenes(
    video_path: str | Path,
    taxonomy: Taxonomy | None = None,
    *,
    log: bool = True,
) -> tuple[list[VisualSpan], dict]:
    """Label every scene with Qwen, then free the model. Returns spans and a report."""
    import gc

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    taxonomy = taxonomy or load_taxonomy()
    max_frames = taxonomy.visual_scene_max_frames
    min_look = taxonomy.visual_scene_min_look_sec
    chunks, duration = scene_chunks(video_path)

    processor = AutoProcessor.from_pretrained(taxonomy.visual_scene_model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        taxonomy.visual_scene_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    spans: list[VisualSpan] = []
    failed: list[dict] = []
    scenes_out: list[dict] = []
    try:
        for i, (start, end) in enumerate(chunks, start=1):
            look_start, look_end = look_window(start, end, duration, min_look_sec=min_look)
            fps = sample_fps(look_start, look_end, max_frames=max_frames, min_look_sec=min_look)
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "video_start": look_start,
                        "video_end": look_end,
                        "fps": fps,
                        "min_frames": 2,
                        "max_frames": max_frames,
                        "cap_pixels_per_frame": True,
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }]
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                if not video_inputs:
                    raise RuntimeError("no frames decoded")
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=80)
                reply = processor.batch_decode(
                    generated[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
            except Exception as exc:
                failed.append({"start_sec": round(start, 3), "end_sec": round(end, 3), "error": str(exc)[:200]})
                if log:
                    print(f"[scene {i}/{len(chunks)}] {start:.1f}-{end:.1f}s FAILED: {exc}")
                continue
            cats = parse_categories(reply)
            scenes_out.append({
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "categories": cats,
                "fps": fps,
            })
            if log:
                print(f"[scene {i}/{len(chunks)}] {start:.1f}-{end:.1f}s -> {', '.join(cats) or 'none'}")
            for cat in cats:
                spans.append(VisualSpan(category=cat, start_sec=start, end_sec=end))
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "model": taxonomy.visual_scene_model,
        "duration_sec": round(duration, 3),
        "scenes": len(chunks),
        "scenes_labeled": len(scenes_out),
        "scenes_failed": failed,
        "per_scene": scenes_out,
    }
    return spans, report
