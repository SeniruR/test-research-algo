"""Regenerate the Colab bootstrap cell from haptic_gt source files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "haptic_gt"
NOTEBOOK = ROOT / "notebooks" / "haptic_groundtruth_colab.ipynb"

BOOTSTRAP_HEADER = '''import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/haptic-groundtruth")
PKG_DIR = PROJECT_ROOT / "haptic_gt"
PKG_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
'''

BOOTSTRAP_FOOTER = '''
}

for name, source in FILES.items():
    target = PKG_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")

sys.path.insert(0, str(PROJECT_ROOT))
print("Installed haptic_gt at", PKG_DIR)
print("Modules:", ", ".join(sorted(FILES)))
'''


def escape_for_triple_quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def collect_modules() -> list[tuple[str, Path]]:
    return sorted(
        (p.relative_to(PKG).as_posix(), p)
        for p in PKG.rglob("*.py")
    )


def main() -> None:
    modules = collect_modules()
    lines = [BOOTSTRAP_HEADER]
    for i, (rel, path) in enumerate(modules):
        content = path.read_text(encoding="utf-8")
        escaped = escape_for_triple_quote(content)
        comma = "," if i < len(modules) - 1 else ""
        lines.append(f'    "{rel}": """{escaped}"""{comma}\n')
    lines.append(BOOTSTRAP_FOOTER)
    bootstrap_source = "".join(lines)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "FILES = {" in src or 'PROJECT_ROOT = Path("/content/haptic-groundtruth")' in src:
                cell["source"] = bootstrap_source.splitlines(keepends=True)
                break
    else:
        raise RuntimeError("Bootstrap cell not found in notebook")

    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated", NOTEBOOK, f"({len(modules)} modules)")


if __name__ == "__main__":
    main()
