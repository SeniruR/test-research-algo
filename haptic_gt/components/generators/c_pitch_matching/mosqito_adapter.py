"""Give Pitch_WebTool.py the MoSQITo loudness call it was written against.

Upstream imports ``mosqito.functions.loudness_zwtv._loudness_zwtv`` and reads
``results["N"]`` and ``results["N_specific"]`` (shape [time, 240]). No MoSQITo
release or commit has that module, and none returns a dict: 0.x ships
``functions/loudness_zwicker``, 1.x ships ``sq_metrics.loudness_zwtv``, which
returns ``(N, N_specific, bark_axis, time_axis)`` with N_specific shaped
[240, time]. So upstream always takes its mean-envelope fallback.

``iso532_loudness(module)`` points the unmodified upstream module at MoSQITo
1.x for the duration of a call, leaving the file itself untouched.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import ModuleType

import numpy as np

_MISSING = object()


def mosqito_installed() -> bool:
    try:
        from mosqito.sq_metrics import loudness_zwtv  # noqa: F401
    except Exception:
        return False
    return True


def loudness_zwtv(signal, fs, field_type="free"):
    from mosqito.sq_metrics import loudness_zwtv as zwtv

    n, n_specific, _bark_axis, _time_axis = zwtv(signal, fs, field_type=field_type)
    return {"N": np.asarray(n), "N_specific": np.asarray(n_specific).T}


@contextlib.contextmanager
def iso532_loudness(module: ModuleType) -> Iterator[None]:
    saved = {
        name: getattr(module, name, _MISSING) for name in ("MOSQITO_AVAILABLE", "loudness_zwtv")
    }
    module.MOSQITO_AVAILABLE = True
    module.loudness_zwtv = loudness_zwtv
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                delattr(module, name)
            else:
                setattr(module, name, value)
