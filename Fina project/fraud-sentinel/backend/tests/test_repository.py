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


if __name__ == "__main__":
    unittest.main()

