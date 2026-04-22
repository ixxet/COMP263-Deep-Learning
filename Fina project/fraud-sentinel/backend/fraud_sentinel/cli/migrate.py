"""Apply Postgres migrations."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import asyncpg

from fraud_sentinel.settings import get_settings


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Apply Fraud Sentinel database migrations.")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "migrations",
    )
    args = parser.parse_args()
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("FRAUD_DATABASE_URL is required")
    conn = await asyncpg.connect(settings.database_url)
    try:
        for path in sorted(args.migrations_dir.glob("*.sql")):
            await conn.execute(path.read_text(encoding="utf-8"))
            print(f"applied {path.name}")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

