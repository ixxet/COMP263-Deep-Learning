from __future__ import annotations

import unittest

from fraud_sentinel.risk import Thresholds, case_status_for_band, risk_band, status_from_review


class RiskTests(unittest.TestCase):
    def test_risk_bands(self) -> None:
        thresholds = Thresholds(
            low_risk_score=0.30,
            high_risk_score=0.70,
            elevated_anomaly_score=0.60,
            high_anomaly_score=0.85,
        )
        self.assertEqual(risk_band(0.10, 0.10, thresholds), "low")
        self.assertEqual(risk_band(0.40, 0.10, thresholds), "uncertain")
        self.assertEqual(risk_band(0.10, 0.70, thresholds), "uncertain")
        self.assertEqual(risk_band(0.71, 0.10, thresholds), "high")
        self.assertEqual(risk_band(0.10, 0.90, thresholds), "high")

    def test_case_status_and_review_mapping(self) -> None:
        self.assertEqual(case_status_for_band("low"), "audit_closed")
        self.assertEqual(case_status_for_band("high"), "pending_review")
        self.assertEqual(status_from_review("approve"), "approved")
        self.assertEqual(status_from_review("escalate"), "escalated")
        self.assertEqual(status_from_review("dismiss"), "dismissed")


if __name__ == "__main__":
    unittest.main()

