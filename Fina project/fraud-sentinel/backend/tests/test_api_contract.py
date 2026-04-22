from __future__ import annotations

import importlib.util
import os
import unittest


RUNTIME_DEPS = ("fastapi", "pandas", "torch", "joblib", "prometheus_fastapi_instrumentator")


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in RUNTIME_DEPS),
    "runtime dependencies are not installed",
)
class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("FRAUD_DATABASE_URL", None)
        os.environ["FRAUD_ALLOW_DEMO_MODEL"] = "true"
        os.environ["FRAUD_BATCH_LIMIT"] = "2"

        from fastapi.testclient import TestClient
        from fraud_sentinel.api.main import app

        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_prediction_case_review_flow(self) -> None:
        response = self.client.post("/v1/predict", json=_transaction(amount=10_000, v14=10, v17=10))

        self.assertEqual(response.status_code, 200)
        prediction = response.json()
        self.assertEqual(prediction["risk_band"], "high")
        self.assertIsNotNone(prediction["case_id"])

        case_id = prediction["case_id"]
        case_response = self.client.get(f"/v1/cases/{case_id}")
        self.assertEqual(case_response.status_code, 200)
        self.assertEqual(case_response.json()["status"], "pending_review")

        review_response = self.client.post(
            f"/v1/cases/{case_id}/review",
            json={"action": "escalate", "reviewer": "api-test", "rationale": "Signals are high."},
        )
        self.assertEqual(review_response.status_code, 200)
        self.assertEqual(review_response.json()["status"], "escalated")

    def test_invalid_prediction_payloads_are_rejected(self) -> None:
        missing = _transaction()
        missing.pop("V28")
        self.assertEqual(self.client.post("/v1/predict", json=missing).status_code, 422)

        extra = _transaction()
        extra["Class"] = 0
        self.assertEqual(self.client.post("/v1/predict", json=extra).status_code, 422)

    def test_batch_upload_accepts_valid_rows_and_rejects_bad_rows(self) -> None:
        csv = _csv([_transaction(amount=10_000, v14=10, v17=10), {**_transaction(), "V28": "bad"}])
        response = self.client.post(
            "/v1/predict/batch",
            files={"file": ("transactions.csv", csv, "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["accepted_rows"], 1)
        self.assertEqual(payload["rejected_rows"], 1)

    def test_batch_limit_is_enforced(self) -> None:
        csv = _csv([_transaction(), _transaction(), _transaction()])
        response = self.client.post(
            "/v1/predict/batch",
            files={"file": ("transactions.csv", csv, "text/csv")},
        )

        self.assertEqual(response.status_code, 413)


def _transaction(amount: float = 129.5, v14: float = 0.0, v17: float = 0.0) -> dict[str, float]:
    payload = {"Time": 0.0, "Amount": amount}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    payload["V14"] = v14
    payload["V17"] = v17
    return payload


def _csv(rows: list[dict]) -> str:
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
