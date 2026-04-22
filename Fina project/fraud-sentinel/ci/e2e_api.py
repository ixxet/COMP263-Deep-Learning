#!/usr/bin/env python3
"""HTTP smoke and E2E checks for Fraud Sentinel.

The script uses only the Python standard library so it can run from a laptop,
inside CI, or against a port-forwarded Kubernetes pod.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any
from urllib import error, request


FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fraud Sentinel HTTP smoke checks.")
    parser.add_argument("--base-url", required=True, help="Base URL for the API service.")
    parser.add_argument(
        "--allow-not-ready",
        action="store_true",
        help="Accept readiness failure when model artifacts have not been trained yet.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail unless /readyz returns 200.",
    )
    parser.add_argument(
        "--require-case",
        action="store_true",
        help="Fail if the high-risk sample does not create a case.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health = http("GET", f"{base_url}/healthz")
    expect(health.status == 200, f"/healthz returned {health.status}")
    expect(health.json()["status"] == "ok", "/healthz payload is not ok")

    metrics = http("GET", f"{base_url}/metrics")
    expect(metrics.status == 200, f"/metrics returned {metrics.status}")
    expect("fraud_model_ready" in metrics.text(), "/metrics is missing fraud_model_ready")

    cases = http("GET", f"{base_url}/v1/cases")
    expect(cases.status == 200, f"/v1/cases returned {cases.status}: {cases.text()}")
    expect(isinstance(cases.json(), list), "/v1/cases did not return a list")

    missing_case = http("GET", f"{base_url}/v1/cases/{uuid.uuid4()}")
    expect(missing_case.status == 404, f"missing case lookup returned {missing_case.status}")

    ready = http("GET", f"{base_url}/readyz")
    if ready.status != 200:
        if args.require_ready or not args.allow_not_ready:
            fail(f"/readyz returned {ready.status}: {ready.text()}")
        payload = ready.json()
        detail = payload.get("detail", {})
        expect(detail.get("db_ready") is True, f"database is not ready: {payload}")
        expect(detail.get("model_ready") is False, f"unexpected readiness payload: {payload}")
        print("smoke passed with model_not_ready=true")
        return 0

    run_prediction_flow(base_url, require_case=args.require_case)
    print("e2e passed with model_ready=true")
    return 0


def run_prediction_flow(base_url: str, *, require_case: bool) -> None:
    missing = transaction()
    missing.pop("V28")
    invalid = http("POST", f"{base_url}/v1/predict", json_body=missing)
    expect(invalid.status == 422, f"missing-column prediction returned {invalid.status}")

    extra = transaction()
    extra["Class"] = 0
    invalid_extra = http("POST", f"{base_url}/v1/predict", json_body=extra)
    expect(invalid_extra.status == 422, f"extra-column prediction returned {invalid_extra.status}")

    prediction = http(
        "POST",
        f"{base_url}/v1/predict",
        json_body=transaction(amount=10_000.0, v14=10.0, v17=10.0),
    )
    expect(prediction.status == 200, f"prediction returned {prediction.status}: {prediction.text()}")
    payload = prediction.json()
    for key in ("prediction_id", "risk_score", "anomaly_score", "risk_band", "model_version"):
        expect(key in payload, f"prediction response missing {key}")
    expect(0.0 <= payload["risk_score"] <= 1.0, "risk_score is outside [0, 1]")
    expect(0.0 <= payload["anomaly_score"] <= 1.0, "anomaly_score is outside [0, 1]")
    expect(payload["risk_band"] in {"low", "uncertain", "high"}, "invalid risk_band")

    batch = multipart_csv(
        [
            transaction(amount=10_000.0, v14=10.0, v17=10.0),
            {**transaction(), "V28": "bad"},
        ]
    )
    batch_result = http(
        "POST",
        f"{base_url}/v1/predict/batch",
        body=batch["body"],
        headers={"content-type": batch["content_type"]},
    )
    expect(batch_result.status == 200, f"batch returned {batch_result.status}: {batch_result.text()}")
    batch_payload = batch_result.json()
    expect(batch_payload["accepted_rows"] >= 1, "batch accepted no rows")
    expect(batch_payload["rejected_rows"] >= 1, "batch rejected no bad rows")

    case_id = payload.get("case_id")
    if not case_id:
        expect(not require_case, "high-risk sample did not create a case")
        return

    case_response = http("GET", f"{base_url}/v1/cases/{case_id}")
    expect(case_response.status == 200, f"case detail returned {case_response.status}")
    case_payload = case_response.json()
    expect(case_payload["case_id"] == case_id, "case detail returned the wrong case")

    review = http(
        "POST",
        f"{base_url}/v1/cases/{case_id}/review",
        json_body={
            "action": "escalate",
            "reviewer": "smoke-test",
            "rationale": "High-risk smoke path requires analyst escalation.",
        },
    )
    expect(review.status == 200, f"review returned {review.status}: {review.text()}")
    expect(review.json()["status"] == "escalated", "review did not escalate the case")

    bad_review = http(
        "POST",
        f"{base_url}/v1/cases/{case_id}/review",
        json_body={"action": "escalate", "reviewer": "", "rationale": "x"},
    )
    expect(bad_review.status == 422, f"bad review returned {bad_review.status}")


def transaction(amount: float = 129.5, v14: float = 0.0, v17: float = 0.0) -> dict[str, Any]:
    payload: dict[str, Any] = {"Time": 0.0, "Amount": amount}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    payload["V14"] = v14
    payload["V17"] = v17
    return payload


def multipart_csv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundary = f"fraud-sentinel-{uuid.uuid4().hex}"
    csv_lines = [",".join(FEATURE_COLUMNS)]
    for row in rows:
        csv_lines.append(",".join(str(row[column]) for column in FEATURE_COLUMNS))
    csv_body = "\n".join(csv_lines) + "\n"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="transactions.csv"\r\n'
        "Content-Type: text/csv\r\n\r\n"
        f"{csv_body}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    return {"body": body, "content_type": f"multipart/form-data; boundary={boundary}"}


def http(
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> HttpResult:
    request_headers = dict(headers or {})
    data = body
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        request_headers["content-type"] = "application/json"
    req = request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with request.urlopen(req, timeout=20) as response:
            return HttpResult(response.status, response.read(), dict(response.headers))
    except error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), dict(exc.headers))
    except error.URLError as exc:
        fail(f"{method} {url} failed: {exc}")


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
