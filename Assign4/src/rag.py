"""Retrieval + generation glue.

The retriever loads the FAISS index and chunk metadata once. The generator
talks to a vLLM OpenAI-compatible endpoint (defaults to http://localhost:8000/v1).

Typical use:
    from src.rag import RAG
    rag = RAG()
    result = rag.answer("What does Raskolnikov mean by 'extraordinary men'?")
    print(result["answer"])
    for c in result["chunks"]:
        print(c["title"], c["score"])
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_META_PATH,
    EMBED_MODEL_NAME,
    FAISS_INDEX_PATH,
    GEN_MAX_TOKENS,
    GEN_TEMPERATURE,
    TOP_K,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)


SYSTEM_PROMPT = (
    "You are a careful research assistant for the works of Fyodor Dostoevsky "
    "and Friedrich Nietzsche. Answer the user's question using ONLY the "
    "passages provided in the CONTEXT block. If the context does not contain "
    "the answer, say so plainly. Do not invent facts, quotes, or citations. "
    "When you draw on a passage, cite it inline using its bracketed number, "
    "e.g. [2]. Keep the answer focused and grounded in the text."
)


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    text: str
    author: str
    title: str
    chunk_index: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": float(self.score),
            "text": self.text,
            "author": self.author,
            "title": self.title,
            "chunk_index": self.chunk_index,
        }


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def _load_chunks() -> list[dict]:
    if not CHUNKS_META_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_META_PATH} not found. Run `python -m src.build_index` first."
        )
    with CHUNKS_META_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_faiss() -> faiss.Index:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{FAISS_INDEX_PATH} not found. Run `python -m src.build_index` first."
        )
    return faiss.read_index(str(FAISS_INDEX_PATH))


class RAG:
    """End-to-end retrieval + generation."""

    def __init__(self) -> None:
        self.index = _load_faiss()
        self.chunks = _load_chunks()
        self.embedder = _load_embedder()
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # ---- retrieval ----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        authors: Iterable[str] | None = None,
    ) -> list[RetrievedChunk]:
        q_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # If filtering, over-fetch then post-filter. Fine for 6 books / few k chunks.
        fetch_k = k if not authors else max(k * 6, k)
        scores, ids = self.index.search(q_vec, fetch_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()

        wanted = set(authors) if authors else None
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores, ids):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if wanted and chunk["author"] not in wanted:
                continue
            results.append(
                RetrievedChunk(
                    rank=len(results) + 1,
                    score=score,
                    text=chunk["text"],
                    author=chunk["author"],
                    title=chunk["title"],
                    chunk_index=chunk["chunk_index"],
                )
            )
            if len(results) >= k:
                break
        return results

    # ---- generation ---------------------------------------------------------
    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for c in chunks:
            header = f"[{c.rank}] {c.author} — {c.title} (chunk {c.chunk_index})"
            blocks.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ("No passages were retrieved, so I cannot answer this question "
                    "from the corpus.")
        context = self._format_context(chunks)
        user_msg = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using only the CONTEXT above. Cite passages as [n]."
        )
        resp = self.client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    # ---- convenience --------------------------------------------------------
    def answer(
        self,
        question: str,
        k: int = TOP_K,
        authors: Iterable[str] | None = None,
    ) -> dict:
        chunks = self.retrieve(question, k=k, authors=authors)
        answer = self.generate(question, chunks)
        return {
            "question": question,
            "answer": answer,
            "chunks": [c.to_dict() for c in chunks],
        }
