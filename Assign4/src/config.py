"""Central configuration for the RAG system.

All paths are relative to the Assign4/ project root.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INDEX_DIR = DATA_DIR / "index"

FAISS_INDEX_PATH = INDEX_DIR / "corpus.faiss"
CHUNKS_META_PATH = INDEX_DIR / "chunks.jsonl"

# ---------------------------------------------------------------------------
# Corpus: Project Gutenberg plain-text URLs
# ---------------------------------------------------------------------------
# Each entry produces one .txt file in data/raw/. The header/footer added by
# Project Gutenberg is stripped during cleaning.
CORPUS = [
    # ---- Fyodor Dostoevsky (Constance Garnett translations) ----
    {
        "id": "dostoevsky_crime_and_punishment",
        "author": "Fyodor Dostoevsky",
        "title": "Crime and Punishment",
        "url": "https://www.gutenberg.org/files/2554/2554-0.txt",
    },
    {
        "id": "dostoevsky_notes_from_underground",
        "author": "Fyodor Dostoevsky",
        "title": "Notes from the Underground",
        "url": "https://www.gutenberg.org/files/600/600-0.txt",
    },
    {
        "id": "dostoevsky_brothers_karamazov",
        "author": "Fyodor Dostoevsky",
        "title": "The Brothers Karamazov",
        "url": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    },
    # ---- Friedrich Nietzsche ----
    {
        "id": "nietzsche_zarathustra",
        "author": "Friedrich Nietzsche",
        "title": "Thus Spake Zarathustra",
        "url": "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
    },
    {
        "id": "nietzsche_beyond_good_and_evil",
        "author": "Friedrich Nietzsche",
        "title": "Beyond Good and Evil",
        "url": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    },
    {
        "id": "nietzsche_antichrist",
        "author": "Friedrich Nietzsche",
        "title": "The Antichrist",
        "url": "https://www.gutenberg.org/cache/epub/19322/pg19322.txt",
    },
]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Choices justified in the report: dense philosophical / narrative prose
# benefits from chunks large enough to hold a complete argument or scene
# beat, but small enough that retrieval stays focused.
CHUNK_SIZE_CHARS = 1000        # ~200 tokens
CHUNK_OVERLAP_CHARS = 200      # 20% overlap preserves context across borders

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
EMBED_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5

# ---------------------------------------------------------------------------
# Generation (vLLM, OpenAI-compatible API)
# ---------------------------------------------------------------------------
# Override via environment variables to keep secrets / hostnames out of code.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM ignores the value
VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
)
GEN_TEMPERATURE = 0.2
GEN_MAX_TOKENS = 512
