"""Pydantic request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    prediction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    risk_band: Literal["low", "uncertain", "high"]
    model_version: str
    case_id: str | None = None


class BatchPredictionResponse(BaseModel):
    accepted_rows: int
    rejected_rows: int
    prediction_ids: list[str]
    case_ids: list[str]


class CaseSummary(BaseModel):
    case_id: str
    prediction_id: str
    status: str
    risk_band: str
    risk_score: float
    anomaly_score: float
    created_at: datetime | None = None
    brief: str | None = None


class CaseDetail(CaseSummary):
    transaction: dict[str, float]
    policy_context: list[dict] = []
    reviews: list[dict] = []
    audit_events: list[dict] = []


class ReviewRequest(BaseModel):
    action: Literal["approve", "escalate", "dismiss"]
    reviewer: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=3, max_length=2000)


class ReviewResponse(BaseModel):
    case_id: str
    status: str

