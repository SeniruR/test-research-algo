"""
Fix the corrupted haptic_groundtruth_colab.ipynb by restoring the notebook from
git (index), then regenerating the bootstrap cell from current source files.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "haptic_groundtruth_colab.ipynb"


def restore_from_git() -> bool:
    """Try to get the last clean committed version of the notebook."""
    try:
        result = subprocess.run(
            ["git", "show", "HEAD:notebooks/haptic_groundtruth_colab.ipynb"],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode == 0:
            NOTEBOOK.write_bytes(result.stdout)
            print("Restored notebook from git HEAD.")
            return True
    except Exception as e:
        print(f"git restore failed: {e}")
    return False


def find_and_fix_bad_escape(raw: str) -> str:
    """
    Scan for the bad escape at the known position and fix it.
    The problem: our StrReplace edits inserted content directly into the JSON
    array lines, leaving raw unescaped backslashes like \n, \", etc. inside
    JSON strings. We find those and fix them.
    """
    # The bad position is around char 113879 (python json error pos).
    # The specific issue: lines that are Python source lines stored as JSON
    # strings but contain raw \n or \" that aren't double-escaped.
    #
    # Strategy: find all "source" arrays in the notebook JSON and for each
    # string element, ensure backslashes are properly escaped.
    # However that would corrupt the valid ones too.
    #
    # Better: find the specific bad line by scanning for invalid escape sequences.
    pos = 113879  # from the error
    # Show context
    ctx_start = max(0, pos - 100)
    ctx_end = min(len(raw), pos + 100)
    print(f"Context around error position {pos}:")
    print(repr(raw[ctx_start:ctx_end]))
    return raw  # don't auto-fix, just show context


def main() -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    print(f"Notebook size: {len(raw)} chars")

    # Try to parse — if it works, nothing to do
    try:
        json.loads(raw)
        print("Notebook JSON is already valid!")
        # Still regenerate bootstrap to be safe
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        find_and_fix_bad_escape(raw)
        print()
        print("Attempting git restore...")
        if restore_from_git():
            try:
                json.loads(NOTEBOOK.read_text(encoding="utf-8"))
                print("Git restored version is valid JSON.")
            except json.JSONDecodeError as e2:
                print(f"Git version also invalid: {e2}")
                sys.exit(1)
        else:
            print("Could not restore from git.")
            sys.exit(1)

    # Now regenerate bootstrap from current sources
    print("\nRegenerating bootstrap cell from current sources...")
    gen = ROOT / "scripts" / "generate_colab_bootstrap.py"
    result = subprocess.run([sys.executable, str(gen)], cwd=ROOT, capture_output=False)
    if result.returncode != 0:
        print("Bootstrap regeneration failed.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
