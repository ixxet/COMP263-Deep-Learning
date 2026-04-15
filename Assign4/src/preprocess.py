"""Strip Gutenberg boilerplate and normalise whitespace.

Reads from data/raw/ and writes cleaned files to data/clean/.

Usage:
    python -m src.preprocess
"""
from __future__ import annotations

import re
from pathlib import Path

from src.config import CLEAN_DIR, CORPUS, RAW_DIR


# Gutenberg files have a header and footer wrapped in *** START / *** END
# markers. The exact wording has shifted over the years, so match loosely.
START_MARK = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
    re.IGNORECASE,
)
END_MARK = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
    re.IGNORECASE,
)
LEGACY_END_MARK = re.compile(
    r"\n\s*End of (the )?Project Gutenberg['’]s .+?\n",
    re.IGNORECASE,
)
PRODUCED_BY_MARK = re.compile(
    r"^\s*Produced by .+?(?=\n\s*\n\s*\n)",
    re.IGNORECASE | re.DOTALL,
)
TRANSCRIBER_NOTE_MARK = re.compile(
    r"\n\s*TRANSCRIBER'S NOTE.*?(?=\n\s*\n\s*(TABLE OF CONTENTS|PREFACE)\s*\n)",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_wrapper(text: str) -> str:
    """Remove the Gutenberg legal preamble and trailing license."""
    text = text.lstrip("\ufeff")
    start = START_MARK.search(text)
    end = END_MARK.search(text)
    if start:
        text = text[start.end():]
    if end:
        # `end` was searched on the original text, so re-search after slicing.
        end2 = END_MARK.search(text)
        if end2:
            text = text[: end2.start()]
    legacy_end = LEGACY_END_MARK.search(text)
    if legacy_end:
        text = text[: legacy_end.start()]
    return text


def normalise(text: str) -> str:
    """Collapse weird whitespace while preserving paragraph breaks."""
    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PRODUCED_BY_MARK.sub("", text)
    text = TRANSCRIBER_NOTE_MARK.sub("\n", text)
    # Some Gutenberg texts use _underscores_ for italics — drop the markers.
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    # Collapse runs of >2 newlines to exactly 2 (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces / tabs within a line.
    text = re.sub(r"[ \t]+", " ", text)
    # Trim trailing whitespace on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip() + "\n"


def clean_one(entry: dict) -> Path:
    raw_path = RAW_DIR / f"{entry['id']}.txt"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run src.fetch_corpus first; missing {raw_path}")
    text = raw_path.read_text(encoding="utf-8")
    text = strip_gutenberg_wrapper(text)
    text = normalise(text)
    out_path = CLEAN_DIR / f"{entry['id']}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main() -> int:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cleaning {len(CORPUS)} files into {CLEAN_DIR}")
    for entry in CORPUS:
        out = clean_one(entry)
        size_kb = out.stat().st_size // 1024
        print(f"  {entry['title']:35s} -> {out.name}  ({size_kb} KB)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
