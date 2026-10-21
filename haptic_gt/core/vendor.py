"""Import unmodified third-party scripts that expect to run from their own folder."""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"


def import_from_dir(directory: str | Path, module_name: str) -> ModuleType:
    """Import ``module_name`` with ``directory`` on ``sys.path``.

    Scripts such as Sound2Hap's use top-level imports of their siblings
    (``from utils.normalization import ...``), so the folder itself has to be
    importable rather than the files being loaded as part of a package.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Third-party folder not found: {directory}")
    entry = str(directory)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return importlib.import_module(module_name)


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Swallow stdout from scripts that print progress on every call."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
