"""Persistence adapters for predictions, cases, reviews, and audit events."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from fraud_sentinel.risk import status_from_review


def utcnow() -> datetime:
    return datetime.now(UTC)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=str)


def _json_loads(payload: Any) -> Any:
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


class Repository(ABC):
    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def create_prediction(
        self,
        transaction: dict[str, float],
        prediction: dict[str, Any],
    ) -> tuple[str, str | None]: ...

    @abstractmethod
    async def list_cases(self, status: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_case(self, case_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def pending_cases(self, limit: int = 25) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def save_case_brief(
        self,
        case_id: str,
        brief: str,
        policy_context: list[dict[str, Any]],
        status: str,
    ) -> None: ...

    @abstractmethod
    async def add_review(
        self,
        case_id: str,
        action: str,
        reviewer: str,
        rationale: str,
    ) -> str: ...

    @abstractmethod
    async def save_agent_run(
        self,
        case_id: str,
        status: str,
        state: dict[str, Any],
        error: str | None = None,
    ) -> str: ...

    @abstractmethod
    async def write_audit(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


class MemoryRepository(Repository):
    def __init__(self) -> None:
        self.transactions: dict[str, dict[str, Any]] = {}
        self.predictions: dict[str, dict[str, Any]] = {}
        self.cases: dict[str, dict[str, Any]] = {}
        self.reviews: list[dict[str, Any]] = []
        self.agent_runs: list[dict[str, Any]] = []
        self.audit_events: list[dict[str, Any]] = []

    async def ping(self) -> bool:
        return True

    async def create_prediction(
        self,
        transaction: dict[str, float],
        prediction: dict[str, Any],
    ) -> tuple[str, str | None]:
        transaction_id = str(uuid.uuid4())
        prediction_id = str(uuid.uuid4())
        now = utcnow()
        self.transactions[transaction_id] = {
            "transaction_id": transaction_id,
            "payload": transaction,
            "created_at": now,
        }
        self.predictions[prediction_id] = {
            "prediction_id": prediction_id,
            "transaction_id": transaction_id,
            "risk_score": prediction["risk_score"],
            "anomaly_score": prediction["anomaly_score"],
            "risk_band": prediction["risk_band"],
            "model_version": prediction["model_version"],
            "created_at": now,
        }
        case_id = None
        if prediction["risk_band"] in {"uncertain", "high"}:
            case_id = str(uuid.uuid4())
            self.cases[case_id] = {
                "case_id": case_id,
                "prediction_id": prediction_id,
                "transaction_id": transaction_id,
                "status": "pending_review",
                "risk_band": prediction["risk_band"],
                "risk_score": prediction["risk_score"],
                "anomaly_score": prediction["anomaly_score"],
                "model_version": prediction["model_version"],
                "brief": None,
                "policy_context": [],
                "created_at": now,
                "updated_at": now,
            }
            await self.write_audit("case", case_id, "case_created", prediction)
        await self.write_audit("prediction", prediction_id, "prediction_created", prediction)
        return prediction_id, case_id

    async def list_cases(self, status: str | None = None) -> list[dict[str, Any]]:
        cases = list(self.cases.values())
        if status:
            cases = [case for case in cases if case["status"] == status]
        return sorted(cases, key=lambda case: case["created_at"], reverse=True)

    async def get_case(self, case_id: str) -> dict[str, Any] | None:
        case = self.cases.get(case_id)
        if not case:
            return None
        prediction = self.predictions[case["prediction_id"]]
        transaction = self.transactions[prediction["transaction_id"]]
        return {
            **case,
            "transaction": transaction["payload"],
            "reviews": [review for review in self.reviews if review["case_id"] == case_id],
            "audit_events": [
                event
                for event in self.audit_events
                if event["entity_id"] in {case_id, case["prediction_id"]}
            ],
        }

    async def pending_cases(self, limit: int = 25) -> list[dict[str, Any]]:
        cases = await self.list_cases(status="pending_review")
        return cases[:limit]

    async def save_case_brief(
        self,
        case_id: str,
        brief: str,
        policy_context: list[dict[str, Any]],
        status: str,
    ) -> None:
        if case_id not in self.cases:
            raise KeyError(case_id)
        self.cases[case_id]["brief"] = brief
        self.cases[case_id]["policy_context"] = policy_context
        self.cases[case_id]["status"] = status
        self.cases[case_id]["updated_at"] = utcnow()
        await self.write_audit(
            "case",
            case_id,
            "case_brief_saved",
            {"status": status, "context_count": len(policy_context)},
        )

    async def add_review(
        self,
        case_id: str,
        action: str,
        reviewer: str,
        rationale: str,
    ) -> str:
        review_id = str(uuid.uuid4())
        status = status_from_review(action)  # type: ignore[arg-type]
        review = {
            "review_id": review_id,
            "case_id": case_id,
            "action": action,
            "reviewer": reviewer,
            "rationale": rationale,
            "created_at": utcnow(),
        }
        self.reviews.append(review)
        if case_id in self.cases:
            self.cases[case_id]["status"] = status
            self.cases[case_id]["updated_at"] = utcnow()
        await self.write_audit("case", case_id, "human_review_added", review)
        return status

    async def save_agent_run(
        self,
        case_id: str,
        status: str,
        state: dict[str, Any],
        error: str | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        self.agent_runs.append(
            {
                "agent_run_id": run_id,
                "case_id": case_id,
                "status": status,
                "state": state,
                "error": error,
                "created_at": utcnow(),
            }
        )
        return run_id

    async def write_audit(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.audit_events.append(
            {
                "audit_event_id": str(uuid.uuid4()),
                "entity_type": entity_type,
                "entity_id": entity_id,
                "event_type": event_type,
                "payload": payload,
                "created_at": utcnow(),
            }
        )


class PostgresRepository(Repository):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.pool = None

    async def connect(self) -> None:
        import asyncpg

        self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=8)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def ping(self) -> bool:
        async with self.pool.acquire() as conn:
            return bool(await conn.fetchval("select true"))

    async def create_prediction(
        self,
        transaction: dict[str, float],
        prediction: dict[str, Any],
    ) -> tuple[str, str | None]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                transaction_id = await conn.fetchval(
                    "insert into transactions (payload) values ($1::jsonb) returning transaction_id",
                    _json_dumps(transaction),
                )
                prediction_id = await conn.fetchval(
                    """
                    insert into predictions
                    (transaction_id, risk_score, anomaly_score, risk_band, model_version)
                    values ($1, $2, $3, $4, $5)
                    returning prediction_id
                    """,
                    transaction_id,
                    prediction["risk_score"],
                    prediction["anomaly_score"],
                    prediction["risk_band"],
                    prediction["model_version"],
                )
                case_id = None
                if prediction["risk_band"] in {"uncertain", "high"}:
                    case_id = await conn.fetchval(
                        """
                        insert into fraud_cases
                        (prediction_id, status, risk_band, risk_score, anomaly_score, model_version)
                        values ($1, 'pending_review', $2, $3, $4, $5)
                        returning case_id
                        """,
                        prediction_id,
                        prediction["risk_band"],
                        prediction["risk_score"],
                        prediction["anomaly_score"],
                        prediction["model_version"],
                    )
                    await conn.execute(
                        """
                        insert into audit_events (entity_type, entity_id, event_type, payload)
                        values ('case', $1, 'case_created', $2::jsonb)
                        """,
                        case_id,
                        _json_dumps(prediction),
                    )
                await conn.execute(
                    """
                    insert into audit_events (entity_type, entity_id, event_type, payload)
                    values ('prediction', $1, 'prediction_created', $2::jsonb)
                    """,
                    prediction_id,
                    _json_dumps(prediction),
                )
                return str(prediction_id), str(case_id) if case_id else None

    async def list_cases(self, status: str | None = None) -> list[dict[str, Any]]:
        where = "where status = $1" if status else ""
        args = [status] if status else []
        rows = await self.pool.fetch(
            f"""
            select case_id::text, prediction_id::text, status, risk_band, risk_score,
                   anomaly_score, model_version, brief, policy_context, created_at, updated_at
            from fraud_cases
            {where}
            order by created_at desc
            limit 100
            """,
            *args,
        )
        return [
            {
                **dict(row),
                "policy_context": _json_loads(row["policy_context"]),
            }
            for row in rows
        ]

    async def get_case(self, case_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select c.case_id::text, c.prediction_id::text, c.status, c.risk_band,
                       c.risk_score, c.anomaly_score, c.model_version, c.brief,
                       coalesce(c.policy_context, '[]'::jsonb) as policy_context,
                       c.created_at, c.updated_at, t.payload as transaction
                from fraud_cases c
                join predictions p on p.prediction_id = c.prediction_id
                join transactions t on t.transaction_id = p.transaction_id
                where c.case_id = $1
                """,
                case_id,
            )
            if not row:
                return None
            reviews = await conn.fetch(
                """
                select review_id::text, action, reviewer, rationale, created_at
                from human_reviews
                where case_id = $1
                order by created_at
                """,
                case_id,
            )
            audit = await conn.fetch(
                """
                select audit_event_id::text, entity_type, entity_id::text, event_type, payload, created_at
                from audit_events
                where entity_id = $1 or entity_id = $2
                order by created_at
                """,
                case_id,
                row["prediction_id"],
            )
            return {
                **dict(row),
                "transaction": _json_loads(row["transaction"]),
                "policy_context": _json_loads(row["policy_context"]),
                "reviews": [dict(item) for item in reviews],
                "audit_events": [
                    {**dict(item), "payload": _json_loads(item["payload"])}
                    for item in audit
                ],
            }

    async def pending_cases(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            select case_id::text, prediction_id::text, status, risk_band, risk_score,
                   anomaly_score, model_version, brief, policy_context, created_at, updated_at
            from fraud_cases
            where status = 'pending_review'
            order by created_at
            limit $1
            """,
            limit,
        )
        return [
            {
                **dict(row),
                "policy_context": _json_loads(row["policy_context"]),
            }
            for row in rows
        ]

    async def save_case_brief(
        self,
        case_id: str,
        brief: str,
        policy_context: list[dict[str, Any]],
        status: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                update fraud_cases
                set brief = $2, policy_context = $3::jsonb, status = $4, updated_at = now()
                where case_id = $1
                """,
                case_id,
                brief,
                _json_dumps(policy_context),
                status,
            )
            await conn.execute(
                """
                insert into audit_events (entity_type, entity_id, event_type, payload)
                values ('case', $1, 'case_brief_saved', $2::jsonb)
                """,
                case_id,
                _json_dumps({"status": status, "context_count": len(policy_context)}),
            )

    async def add_review(
        self,
        case_id: str,
        action: str,
        reviewer: str,
        rationale: str,
    ) -> str:
        status = status_from_review(action)  # type: ignore[arg-type]
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    insert into human_reviews (case_id, action, reviewer, rationale)
                    values ($1, $2, $3, $4)
                    """,
                    case_id,
                    action,
                    reviewer,
                    rationale,
                )
                await conn.execute(
                    "update fraud_cases set status = $2, updated_at = now() where case_id = $1",
                    case_id,
                    status,
                )
                await conn.execute(
                    """
                    insert into audit_events (entity_type, entity_id, event_type, payload)
                    values ('case', $1, 'human_review_added', $2::jsonb)
                    """,
                    case_id,
                    _json_dumps({"action": action, "reviewer": reviewer, "rationale": rationale}),
                )
        return status

    async def save_agent_run(
        self,
        case_id: str,
        status: str,
        state: dict[str, Any],
        error: str | None = None,
    ) -> str:
        async with self.pool.acquire() as conn:
            run_id = await conn.fetchval(
                """
                insert into agent_runs (case_id, status, state, error)
                values ($1, $2, $3::jsonb, $4)
                returning agent_run_id
                """,
                case_id,
                status,
                _json_dumps(state),
                error,
            )
            return str(run_id)

    async def write_audit(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                insert into audit_events (entity_type, entity_id, event_type, payload)
                values ($1, $2, $3, $4::jsonb)
                """,
                entity_type,
                entity_id,
                event_type,
                _json_dumps(payload),
            )
