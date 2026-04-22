"""Risk banding and review gate policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskBand = Literal["low", "uncertain", "high"]
CaseStatus = Literal["audit_closed", "pending_review", "approved", "escalated", "dismissed"]
ReviewAction = Literal["approve", "escalate", "dismiss"]


@dataclass(frozen=True)
class Thresholds:
    low_risk_score: float = 0.35
    high_risk_score: float = 0.70
    elevated_anomaly_score: float = 0.65
    high_anomaly_score: float = 0.85

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "Thresholds":
        if not raw:
            return cls()
        return cls(
            low_risk_score=float(raw.get("low_risk_score", cls.low_risk_score)),
            high_risk_score=float(raw.get("high_risk_score", cls.high_risk_score)),
            elevated_anomaly_score=float(
                raw.get("elevated_anomaly_score", cls.elevated_anomaly_score)
            ),
            high_anomaly_score=float(raw.get("high_anomaly_score", cls.high_anomaly_score)),
        )


def risk_band(risk_score: float, anomaly_score: float, thresholds: Thresholds) -> RiskBand:
    if risk_score >= thresholds.high_risk_score or anomaly_score >= thresholds.high_anomaly_score:
        return "high"
    if (
        risk_score >= thresholds.low_risk_score
        or anomaly_score >= thresholds.elevated_anomaly_score
    ):
        return "uncertain"
    return "low"


def case_status_for_band(band: RiskBand) -> CaseStatus:
    return "audit_closed" if band == "low" else "pending_review"


def requires_human_review(band: RiskBand) -> bool:
    return band in {"uncertain", "high"}


def status_from_review(action: ReviewAction) -> CaseStatus:
    return {
        "approve": "approved",
        "escalate": "escalated",
        "dismiss": "dismissed",
    }[action]

