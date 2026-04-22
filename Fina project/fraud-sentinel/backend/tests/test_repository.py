from __future__ import annotations

import unittest

from fraud_sentinel.feature_schema import FEATURE_COLUMNS
from fraud_sentinel.repository import MemoryRepository


class MemoryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_prediction_creates_case_for_high_risk(self) -> None:
        repo = MemoryRepository()
        transaction = {column: 0.0 for column in FEATURE_COLUMNS}
        transaction["Amount"] = 500.0
        prediction_id, case_id = await repo.create_prediction(
            transaction,
            {
                "risk_score": 0.91,
                "anomaly_score": 0.40,
                "risk_band": "high",
                "model_version": "test-model",
            },
        )
        self.assertIsNotNone(prediction_id)
        self.assertIsNotNone(case_id)
        case = await repo.get_case(case_id or "")
        self.assertIsNotNone(case)
        self.assertEqual(case["status"], "pending_review")

    async def test_review_updates_case_status(self) -> None:
        repo = MemoryRepository()
        transaction = {column: 0.0 for column in FEATURE_COLUMNS}
        _, case_id = await repo.create_prediction(
            transaction,
            {
                "risk_score": 0.80,
                "anomaly_score": 0.20,
                "risk_band": "high",
                "model_version": "test-model",
            },
        )
        status = await repo.add_review(case_id or "", "escalate", "analyst", "Both signals are high.")
        self.assertEqual(status, "escalated")
        case = await repo.get_case(case_id or "")
        self.assertEqual(case["status"], "escalated")

    async def test_low_risk_prediction_does_not_create_case(self) -> None:
        repo = MemoryRepository()
        transaction = {column: 0.0 for column in FEATURE_COLUMNS}
        prediction_id, case_id = await repo.create_prediction(
            transaction,
            {
                "risk_score": 0.05,
                "anomaly_score": 0.02,
                "risk_band": "low",
                "model_version": "test-model",
            },
        )

        self.assertIsNotNone(prediction_id)
        self.assertIsNone(case_id)
        self.assertEqual(await repo.list_cases(), [])
        self.assertTrue(any(event["event_type"] == "prediction_created" for event in repo.audit_events))

    async def test_pending_cases_respects_limit(self) -> None:
        repo = MemoryRepository()
        transaction = {column: 0.0 for column in FEATURE_COLUMNS}
        for index in range(3):
            transaction["Amount"] = float(index)
            await repo.create_prediction(
                dict(transaction),
                {
                    "risk_score": 0.90,
                    "anomaly_score": 0.40,
                    "risk_band": "high",
                    "model_version": "test-model",
                },
            )

        pending = await repo.pending_cases(limit=2)

        self.assertEqual(len(pending), 2)

    async def test_save_missing_case_brief_fails(self) -> None:
        repo = MemoryRepository()
        with self.assertRaises(KeyError):
            await repo.save_case_brief("missing", "brief", [], "awaiting_human_review")


if __name__ == "__main__":
    unittest.main()
