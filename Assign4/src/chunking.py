"""Recursive character splitter (LangChain-style, no LangChain dependency).

Splits text on a priority list of separators (paragraph -> sentence -> word),
producing chunks no larger than ``chunk_size`` characters with a sliding
``chunk_overlap`` between consecutive chunks. The recursive strategy keeps
chunk boundaries on natural breakpoints whenever possible.
"""
from __future__ import annotations

from typing import List


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_with_overlap(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Pack already-small pieces into <= chunk_size windows with overlap."""
    if not pieces:
        return []
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Start the next window with the tail of the previous one for overlap.
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                current = (tail + " " + piece).strip()
            else:
                current = piece
            # If a single piece is larger than chunk_size, hard-split it.
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                tail = current[chunk_size - overlap : chunk_size] if overlap else ""
                current = tail + current[chunk_size:]
    if current:
        chunks.append(current)
    return chunks


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str] | None = None,
) -> List[str]:
    """Split ``text`` recursively, descending the separator list as needed."""
    seps = separators or DEFAULT_SEPARATORS
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = seps[0]
    if sep == "":
        # Last-resort: hard split on character count.
        return _split_with_overlap(
            [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)],
            chunk_size,
            chunk_overlap,
        )

    pieces = text.split(sep)
    # Re-attach the separator (except for "\n\n"/"\n", which are kept implicitly
    # by joining with " " inside the packing step).
    if sep not in {"\n\n", "\n"}:
        pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]

    # Any piece still too large gets split with the next separator.
    expanded: List[str] = []
    for piece in pieces:
        if len(piece) > chunk_size:
            expanded.extend(
                recursive_split(piece, chunk_size, chunk_overlap, seps[1:])
            )
        else:
            expanded.append(piece)

    return _split_with_overlap(expanded, chunk_size, chunk_overlap)
