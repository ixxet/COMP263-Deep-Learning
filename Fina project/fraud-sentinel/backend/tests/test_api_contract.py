from __future__ import annotations

import csv
import importlib.util
import os
import unittest
from io import StringIO


RUNTIME_DEPS = (
    "fastapi",
    "joblib",
    "multipart",
    "pandas",
    "prometheus_fastapi_instrumentator",
    "pydantic_settings",
    "torch",
)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in RUNTIME_DEPS),
    "runtime dependencies are not installed",
)
class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.pop("FRAUD_DATABASE_URL", None)
        os.environ["FRAUD_ALLOW_DEMO_MODEL"] = "true"
        os.environ["FRAUD_BATCH_LIMIT"] = "20"

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
        csv = _csv(
            [_transaction(amount=10_000, v14=10, v17=10), {**_transaction(), "V28": "bad"}],
            include_label=True,
        )
        response = self.client.post(
            "/v1/predict/batch",
            files={"file": ("transactions.csv", csv, "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["accepted_rows"], 1)
        self.assertEqual(payload["rejected_rows"], 1)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["row_index"], 1)
        self.assertEqual(payload["rows"][0]["risk_band"], "high")
        self.assertEqual(len(payload["rejections"]), 1)
        self.assertEqual(payload["rejections"][0]["row_index"], 2)
        self.assertIn("V28 must be numeric", payload["rejections"][0]["reason"])

    def test_prediction_history_includes_low_risk_audit_records(self) -> None:
        response = self.client.post("/v1/predict", json=_transaction(amount=10))
        self.assertEqual(response.status_code, 200)
        prediction = response.json()
        self.assertEqual(prediction["risk_band"], "low")
        self.assertIsNone(prediction["case_id"])

        history_response = self.client.get("/v1/predictions?risk_band=low&has_case=false&limit=10")
        self.assertEqual(history_response.status_code, 200)
        history = history_response.json()
        self.assertTrue(any(row["prediction_id"] == prediction["prediction_id"] for row in history))
        row = next(row for row in history if row["prediction_id"] == prediction["prediction_id"])
        self.assertIsNone(row["case_id"])
        self.assertEqual(row["amount"], 10)

    def test_demo_csv_download_matches_batch_schema(self) -> None:
        response = self.client.get("/v1/samples/demo.csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/csv")

        rows = list(csv.DictReader(StringIO(response.text)))
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(set(rows[0].keys()), {"Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"})

        batch_response = self.client.post(
            "/v1/predict/batch",
            files={"file": ("demo.csv", response.text, "text/csv")},
        )
        self.assertEqual(batch_response.status_code, 200)
        payload = batch_response.json()
        self.assertEqual(payload["accepted_rows"], len(rows))
        self.assertEqual(payload["rejected_rows"], 0)
        self.assertTrue({"low", "uncertain", "high"}.issubset({row["risk_band"] for row in payload["rows"]}))

    def test_batch_limit_is_enforced(self) -> None:
        csv = _csv([_transaction() for _ in range(21)])
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


def _csv(rows: list[dict], *, include_label: bool = False) -> str:
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    if include_label:
        columns.append("Class")
    lines = [",".join(columns)]
    for row in rows:
        if include_label and "Class" not in row:
            row = {**row, "Class": 0}
        lines.append(",".join(str(row[column]) for column in columns))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
