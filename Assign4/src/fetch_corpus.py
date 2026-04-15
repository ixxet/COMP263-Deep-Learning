"""Download the corpus from Project Gutenberg into data/raw/.

Usage:
    python -m src.fetch_corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

from src.config import CORPUS, RAW_DIR


USER_AGENT = "Centennial-DL-263-Assign4/1.0 (educational use)"


def download_one(entry: dict, raw_dir: Path) -> Path:
    out_path = raw_dir / f"{entry['id']}.txt"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] {out_path.name} already present "
              f"({out_path.stat().st_size // 1024} KB)")
        return out_path

    print(f"  [get ] {entry['title']} -> {out_path.name}")
    resp = requests.get(entry["url"], headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    # Project Gutenberg files are typically UTF-8; fall back to latin-1.
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1")
    out_path.write_text(text, encoding="utf-8")
    print(f"         saved {out_path.stat().st_size // 1024} KB")
    return out_path


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CORPUS)} works to {RAW_DIR}")
    failures = 0
    for entry in CORPUS:
        try:
            download_one(entry, RAW_DIR)
        except requests.RequestException as exc:
            print(f"  [fail] {entry['title']}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"\n{failures} download(s) failed.", file=sys.stderr)
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
