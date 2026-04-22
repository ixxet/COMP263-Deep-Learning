"""Process pending fraud cases through the LangGraph workflow."""

from __future__ import annotations

import argparse
import asyncio
import logging

from fraud_sentinel.agent.graph import process_pending_cases
from fraud_sentinel.repository import PostgresRepository
from fraud_sentinel.settings import get_settings


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Run Fraud Sentinel agent worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("FRAUD_DATABASE_URL is required for the deployed agent worker")
    repo = PostgresRepository(settings.database_url)
    await repo.connect()
    try:
        while True:
            processed = await process_pending_cases(repo, settings, limit=args.limit)
            logging.info("processed %s pending cases", processed)
            if args.once:
                break
            await asyncio.sleep(args.interval)
    finally:
        await repo.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(amain())


if __name__ == "__main__":
    main()

