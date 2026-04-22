"""Seed Qdrant with local policy documents using TEI embeddings."""

from __future__ import annotations

import argparse
import asyncio

import httpx

from fraud_sentinel.agent.rag import embed_text, load_policy_documents
from fraud_sentinel.settings import get_settings


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Seed Fraud Sentinel policy corpus.")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.qdrant_url or not settings.tei_base_url:
        raise RuntimeError("FRAUD_QDRANT_URL and FRAUD_TEI_BASE_URL are required")

    docs = load_policy_documents(settings.policy_dir)
    if not docs:
        raise RuntimeError(f"no policy documents found in {settings.policy_dir}")
    vectors = [await embed_text(settings.tei_base_url, doc["content"]) for doc in docs]
    vector_size = len(vectors[0])

    async with httpx.AsyncClient(timeout=30) as client:
        if args.recreate:
            await client.delete(
                f"{settings.qdrant_url}/collections/{settings.qdrant_collection}"
            )
        await client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        points = [
            {
                "id": doc["id"],
                "vector": vector,
                "payload": {
                    "title": doc["title"],
                    "source": doc["source"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                },
            }
            for doc, vector in zip(docs, vectors)
        ]
        response = await client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points",
            json={"points": points},
        )
        response.raise_for_status()
    print(f"seeded {len(docs)} policy documents")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
