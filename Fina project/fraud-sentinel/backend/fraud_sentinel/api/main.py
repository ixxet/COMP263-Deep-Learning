"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import StringIO
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_sentinel.agent.graph import CaseReviewService
from fraud_sentinel.feature_schema import coerce_transaction
from fraud_sentinel.model.artifacts import DemoModelBundle, ModelBundle
from fraud_sentinel.repository import MemoryRepository, PostgresRepository, Repository
from fraud_sentinel.schemas import (
    BatchPredictionResponse,
    CaseDetail,
    CaseSummary,
    PredictionResponse,
    ReviewRequest,
    ReviewResponse,
    TransactionInput,
)
from fraud_sentinel.settings import Settings, get_settings

PREDICTIONS = Counter("fraud_predictions_total", "Predictions by risk band", ["risk_band"])
CASES = Counter("fraud_cases_created_total", "Cases created by risk band", ["risk_band"])
MODEL_READY = Gauge("fraud_model_ready", "Whether model artifacts are loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.repository = await _build_repository(settings)
    app.state.model_error = None
    try:
        app.state.model = ModelBundle.load(settings.model_dir)
        MODEL_READY.set(1)
    except Exception as exc:
        app.state.model_error = str(exc)
        MODEL_READY.set(0)
        if settings.allow_demo_model:
            app.state.model = DemoModelBundle()
        else:
            app.state.model = None
    yield
    repo = app.state.repository
    if isinstance(repo, PostgresRepository):
        await repo.close()


app = FastAPI(
    title="Fraud Sentinel API",
    version="0.1.0",
    description="Credit-card fraud detection API with case review orchestration.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


async def _build_repository(settings: Settings) -> Repository:
    if settings.database_url:
        repo = PostgresRepository(settings.database_url)
        await repo.connect()
        return repo
    return MemoryRepository()


def get_repo() -> Repository:
    return app.state.repository


def get_model():
    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"model artifact bundle is not ready: {app.state.model_error}",
        )
    return model


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(repo: Repository = Depends(get_repo)) -> dict[str, Any]:
    db_ready = await repo.ping()
    model_ready = app.state.model is not None
    if not db_ready or not model_ready:
        raise HTTPException(
            status_code=503,
            detail={"db_ready": db_ready, "model_ready": model_ready, "model_error": app.state.model_error},
        )
    return {"db_ready": db_ready, "model_ready": model_ready}


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    payload: TransactionInput,
    repo: Repository = Depends(get_repo),
    model=Depends(get_model),
) -> PredictionResponse:
    transaction = coerce_transaction(payload.model_dump())
    model_prediction = model.predict(transaction)
    prediction_data = {
        "risk_score": model_prediction.risk_score,
        "anomaly_score": model_prediction.anomaly_score,
        "risk_band": model_prediction.risk_band,
        "model_version": model_prediction.model_version,
    }
    prediction_id, case_id = await repo.create_prediction(transaction, prediction_data)
    PREDICTIONS.labels(risk_band=model_prediction.risk_band).inc()
    if case_id:
        CASES.labels(risk_band=model_prediction.risk_band).inc()
    return PredictionResponse(prediction_id=prediction_id, case_id=case_id, **prediction_data)


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    repo: Repository = Depends(get_repo),
    model=Depends(get_model),
) -> BatchPredictionResponse:
    settings: Settings = app.state.settings
    raw = await file.read()
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid CSV: {exc}") from exc
    if len(df) > settings.batch_limit:
        raise HTTPException(status_code=413, detail=f"batch limit is {settings.batch_limit} rows")

    prediction_ids: list[str] = []
    case_ids: list[str] = []
    rejected = 0
    for record in df.to_dict(orient="records"):
        try:
            transaction = coerce_transaction(record, allow_label=True)
            model_prediction = model.predict(transaction)
            prediction_data = {
                "risk_score": model_prediction.risk_score,
                "anomaly_score": model_prediction.anomaly_score,
                "risk_band": model_prediction.risk_band,
                "model_version": model_prediction.model_version,
            }
            prediction_id, case_id = await repo.create_prediction(transaction, prediction_data)
        except Exception:
            rejected += 1
            continue
        prediction_ids.append(prediction_id)
        if case_id:
            case_ids.append(case_id)
        PREDICTIONS.labels(risk_band=model_prediction.risk_band).inc()

    return BatchPredictionResponse(
        accepted_rows=len(prediction_ids),
        rejected_rows=rejected,
        prediction_ids=prediction_ids,
        case_ids=case_ids,
    )


@app.get("/v1/cases", response_model=list[CaseSummary])
async def list_cases(
    status: str | None = None,
    repo: Repository = Depends(get_repo),
) -> list[dict[str, Any]]:
    return await repo.list_cases(status=status)


@app.get("/v1/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str, repo: Repository = Depends(get_repo)) -> dict[str, Any]:
    case = await repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@app.post("/v1/cases/{case_id}/review", response_model=ReviewResponse)
async def review_case(
    case_id: str,
    payload: ReviewRequest,
    repo: Repository = Depends(get_repo),
) -> ReviewResponse:
    case = await repo.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    status = await repo.add_review(
        case_id=case_id,
        action=payload.action,
        reviewer=payload.reviewer,
        rationale=payload.rationale,
    )
    service = CaseReviewService(repo, app.state.settings)
    try:
        await service.resume_case(case_id, payload.model_dump())
    except Exception as exc:
        await repo.save_agent_run(
            case_id,
            "resume_skipped",
            {"case_id": case_id, "review": payload.model_dump()},
            str(exc),
        )
    finally:
        service.close()
    return ReviewResponse(case_id=case_id, status=status)
