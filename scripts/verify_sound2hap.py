"""Check that third_party/sound2hap holds the unmodified upstream Sound2Hap files.

    python scripts/verify_sound2hap.py              # local files vs SHA256SUMS
    python scripts/verify_sound2hap.py --upstream   # also vs GitHub at the pinned commit
    python scripts/verify_sound2hap.py --download   # (re)fetch the files from GitHub

Anyone can also check by hand: ``sha256sum -c SHA256SUMS`` inside
third_party/sound2hap, or delete the folder's .py files and download them from
https://github.com/Iris1215/Sound2Hap at the commit named in UPSTREAM.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "third_party" / "sound2hap"
SUMS_FILE = VENDOR_DIR / "SHA256SUMS"

REPO = "Iris1215/Sound2Hap"
COMMIT = "0af5934173ebd9cb35eed2d63fbdb79ac75a75d8"
#: local path (inside VENDOR_DIR) -> path in the upstream repository
FILES = {
    "FreqShift.py": "Signal_Processing_Algorithms/FreqShift.py",
    "HapticGen.py": "Signal_Processing_Algorithms/HapticGen.py",
    "Percept.py": "Signal_Processing_Algorithms/Percept.py",
    "Pitch_WebTool.py": "Signal_Processing_Algorithms/Pitch_WebTool.py",
    "utils/normalization.py": "Signal_Processing_Algorithms/utils/normalization.py",
    "LICENSE": "LICENSE",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(upstream_path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{upstream_path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _read_sums() -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in SUMS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def download() -> None:
    lines = []
    for local, upstream in FILES.items():
        data = _fetch(upstream)
        target = VENDOR_DIR / local
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        lines.append(f"{_sha256(data)}  {local}")
        print("fetched", local)
    SUMS_FILE.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("wrote", SUMS_FILE)


def verify(upstream: bool) -> int:
    sums = _read_sums()
    failures = 0
    for local, upstream_path in FILES.items():
        path = VENDOR_DIR / local
        expected = sums.get(local)
        if not path.is_file():
            print(f"MISSING   {local}")
            failures += 1
            continue
        actual = _sha256(path.read_bytes())
        status = "OK" if actual == expected else "MODIFIED"
        if upstream:
            remote = _sha256(_fetch(upstream_path))
            if remote != actual:
                status = "DIFFERS FROM GITHUB"
        if status != "OK":
            failures += 1
        print(f"{status:<9} {actual[:16]}  {local}")
    if failures:
        print(f"{failures} file(s) do not match upstream {REPO}@{COMMIT[:7]}")
        return 1
    where = "SHA256SUMS and GitHub" if upstream else "SHA256SUMS"
    print(f"all Sound2Hap files match {where} ({REPO}@{COMMIT[:7]})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", action="store_true", help="also compare against GitHub")
    parser.add_argument("--download", action="store_true", help="fetch the files from GitHub")
    args = parser.parse_args()
    if args.download:
        download()
    return verify(args.upstream)


if __name__ == "__main__":
    sys.exit(main())
