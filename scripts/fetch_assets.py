"""Download and SHA-256-verify the pinned model checkpoint and demo images.

Usage: python scripts/fetch_assets.py
Everything listed in scripts/assets.json is fetched once; files already present
with the right hash are left alone, and a hash mismatch is a hard error.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str) -> None:
    if dest.exists() and sha256_of(dest) == expected_sha256:
        print(f"ok (cached)  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading  {url}")
    # curl uses the macOS trust store; python.org Python builds ship without CA
    # certificates unless "Install Certificates.command" was run.
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "-o", str(partial), url],
                   check=True)
    actual = sha256_of(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise SystemExit(f"SHA-256 mismatch for {url}\n  expected {expected_sha256}\n  actual   {actual}")
    partial.replace(dest)
    print(f"ok (verified) {dest.relative_to(ROOT)}")


def main() -> int:
    assets = json.loads((ROOT / "scripts" / "assets.json").read_text())

    ckpt = assets["checkpoint"]
    fetch(ckpt["url"], ROOT / ckpt["destination"], ckpt["sha256"])

    demos = assets["demo_images"]
    for name, digest in demos["files"].items():
        fetch(demos["base_url"] + name, ROOT / demos["destination"] / name, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
