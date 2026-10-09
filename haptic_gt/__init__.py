"""Ground-truth haptic track generation from video audio."""

from __future__ import annotations

__all__ = ["load_pipeline_config", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from haptic_gt.core.runner import run_pipeline

        return run_pipeline
    if name == "load_pipeline_config":
        from haptic_gt.core.config import load_pipeline_config

        return load_pipeline_config
    raise AttributeError(f"module 'haptic_gt' has no attribute {name!r}")
