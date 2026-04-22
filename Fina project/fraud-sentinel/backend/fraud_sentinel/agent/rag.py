"""Policy retrieval and vLLM generation helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from fraud_sentinel.agent.prompts import SYSTEM_PROMPT, build_case_prompt
from fraud_sentinel.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = PROJECT_ROOT / "policy"


def load_policy_documents(policy_dir: Path = POLICY_DIR) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(policy_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        title = content.splitlines()[0].lstrip("# ").strip() if content else path.stem
        docs.append(
            {
                "id": hashlib.sha256(str(path).encode()).hexdigest()[:16],
                "source": str(path.name),
                "title": title,
                "content": content,
                "metadata": {"path": str(path)},
            }
        )
    return docs


class PolicyRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_docs = load_policy_documents(settings.policy_dir)

    async def retrieve(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if self.settings.qdrant_url and self.settings.tei_base_url:
            try:
                return await self._retrieve_qdrant(query, limit=limit)
            except Exception:
                pass
        return self._retrieve_local(query, limit=limit)

    async def _retrieve_qdrant(self, query: str, limit: int) -> list[dict[str, Any]]:
        vector = await embed_text(self.settings.tei_base_url, query)
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.settings.qdrant_url}/collections/{self.settings.qdrant_collection}/points/search",
                json={"vector": vector, "limit": limit, "with_payload": True},
            )
            response.raise_for_status()
        hits = response.json().get("result", [])
        return [
            {
                "title": hit.get("payload", {}).get("title", "Policy document"),
                "content": hit.get("payload", {}).get("content", ""),
                "source": hit.get("payload", {}).get("source", "qdrant"),
                "score": hit.get("score", 0.0),
            }
            for hit in hits
        ]

    def _retrieve_local(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = {term.lower() for term in query.replace("_", " ").split() if len(term) > 3}
        ranked = []
        for doc in self.local_docs:
            text = f"{doc['title']} {doc['content']}".lower()
            score = sum(1 for term in terms if term in text)
            ranked.append(({**doc, "score": float(score)}, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in ranked[:limit]]


async def embed_text(tei_base_url: str, text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{tei_base_url.rstrip('/')}/embed",
            json={"inputs": text},
        )
        response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return payload[0]
    if isinstance(payload, dict) and "embedding" in payload:
        return payload["embedding"]
    raise ValueError("TEI response did not contain an embedding")


class BriefGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )

    async def generate(self, case: dict, context: list[dict]) -> str:
        prompt = build_case_prompt(case, context)

        def _call() -> str:
            response = self.client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()

        import asyncio

        return await asyncio.to_thread(_call)


def deterministic_brief(case: dict, context: list[dict]) -> str:
    citation = "[1]" if context else ""
    return (
        f"Case {case['case_id']} entered review because it is {case['risk_band']} "
        f"with risk score {case['risk_score']:.3f} and anomaly score "
        f"{case['anomaly_score']:.3f}. The model can prioritize review, but it "
        f"does not prove fraud {citation}. Recommended analyst action: escalate "
        "when both signals are elevated; otherwise inspect operational context "
        f"before approving or dismissing {citation}."
    )
