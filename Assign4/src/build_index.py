"""Chunk the cleaned corpus, embed with MiniLM, and persist a FAISS index.

Usage:
    python -m src.build_index
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.chunking import recursive_split
from src.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    CHUNKS_META_PATH,
    CLEAN_DIR,
    CORPUS,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
)


def load_chunks() -> list[dict]:
    """Return [{id, author, title, doc_id, chunk_index, text}, ...]."""
    chunks: list[dict] = []
    next_id = 0
    for entry in CORPUS:
        path = CLEAN_DIR / f"{entry['id']}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run src.fetch_corpus then src.preprocess first."
            )
        text = path.read_text(encoding="utf-8")
        pieces = recursive_split(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": next_id,
                "doc_id": entry["id"],
                "author": entry["author"],
                "title": entry["title"],
                "chunk_index": i,
                "text": piece,
            })
            next_id += 1
        print(f"  {entry['title']:35s} -> {len(pieces):4d} chunks")
    print(f"Total chunks: {len(chunks)}")
    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch={EMBED_BATCH_SIZE})")
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via inner product
    )
    return vecs.astype(np.float32)


def build_faiss(vecs: np.ndarray) -> faiss.Index:
    dim = vecs.shape[1]
    # Inner-product index on L2-normalised vectors == cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def save(index: faiss.Index, chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with CHUNKS_META_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote FAISS index -> {FAISS_INDEX_PATH}")
    print(f"Wrote chunk metadata -> {CHUNKS_META_PATH}")


def main() -> int:
    print("=== Chunking ===")
    chunks = load_chunks()
    print("\n=== Embedding ===")
    vecs = embed_chunks(chunks)
    print("\n=== FAISS ===")
    index = build_faiss(vecs)
    print(f"Index size: {index.ntotal} vectors x {vecs.shape[1]} dims")
    save(index, chunks)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
