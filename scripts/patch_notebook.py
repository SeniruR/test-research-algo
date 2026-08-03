"""
Patch haptic_groundtruth_colab.ipynb bootstrap cell with current source files.
Run: python scripts/patch_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "haptic_gt"
NOTEBOOK = ROOT / "notebooks" / "haptic_groundtruth_colab.ipynb"

# Files to patch inside the bootstrap cell FILES dict
PATCH_FILES = [
    "context/taxonomy.yaml",
    "context/taxonomy.py",
    "context/context_detectors.py",
    "context/event_aggregation.py",
    "context/frozen_fusion.py",
    "context/onset_refine.py",
    "haptic_synthesis.py",
]

def _src(rel: str) -> str:
    path = PKG / rel
    return path.read_text(encoding="utf-8")


def _escape_for_triple_quoted(text: str) -> str:
    """Escape backslashes and triple-quotes so text is safe inside Python \"\"\".\"\"\" strings."""
    # In the notebook, triple-quoted Python strings inside JSON source arrays are stored as:
    # each line is a separate JSON array element string.
    # We don't need to touch anything — we just embed the raw source lines.
    return text


def patch_bootstrap_cell(nb: dict) -> bool:
    """Find the bootstrap cell and replace each patched file's source lines."""
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src_lines: list[str] = cell["source"]
        # Detect bootstrap cell by presence of FILES = { marker
        joined = "".join(src_lines)
        if "FILES = {" not in joined or "haptic_synthesis.py" not in joined:
            continue

        # Rebuild the source lines for this cell
        new_lines: list[str] = []
        i = 0
        while i < len(src_lines):
            line = src_lines[i]
            # Detect start of a file block: '    "some/path.py": """'  (after JSON decode)
            # In the raw list, the line looks like: '    "context/taxonomy.yaml": """\n'
            matched_file = None
            for rel in PATCH_FILES:
                marker = f'    "{rel}": """'
                if line.startswith(marker):
                    matched_file = rel
                    break

            if matched_file is None:
                new_lines.append(line)
                i += 1
                continue

            # Found a file block — keep the opening line as-is
            new_lines.append(line)
            i += 1

            # Skip old content until we hit the closing '""",\n' line
            while i < len(src_lines):
                if src_lines[i].rstrip("\n") == '""",':
                    break
                i += 1

            # Insert new file content as individual lines
            content = _src(matched_file)
            for content_line in content.splitlines(keepends=True):
                # Each content line must be a plain string (no JSON escaping needed here
                # because we're building the Python list, not raw JSON)
                new_lines.append(content_line)

            # Append closing delimiter
            new_lines.append('""",\n')
            i += 1  # skip the old closing line
            changed = True
            print(f"  Patched: {matched_file}")

        cell["source"] = new_lines
        break  # only one bootstrap cell

    return changed


def main() -> None:
    print(f"Reading {NOTEBOOK}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    print("Patching bootstrap cell...")
    changed = patch_bootstrap_cell(nb)

    if not changed:
        print("Nothing patched — bootstrap cell format may differ.")
        return

    out = json.dumps(nb, indent=1, ensure_ascii=False)
    NOTEBOOK.write_text(out, encoding="utf-8")
    print(f"Written {NOTEBOOK}")

    # Validate
    try:
        json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        print("JSON validation: OK")
    except json.JSONDecodeError as e:
        print(f"JSON validation FAILED: {e}")


if __name__ == "__main__":
    main()
