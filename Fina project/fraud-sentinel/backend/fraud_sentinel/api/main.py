"""FastAPI entrypoint."""

from __future__ import annotations

import csv
from contextlib import asynccontextmanager
from io import StringIO
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_sentinel.agent.graph import CaseReviewService
from fraud_sentinel.feature_schema import FEATURE_COLUMNS, LABEL_COLUMN, coerce_transaction
from fraud_sentinel.model.artifacts import DemoModelBundle, ModelBundle
from fraud_sentinel.repository import MemoryRepository, PostgresRepository, Repository
from fraud_sentinel.schemas import (
    BatchPredictionRow,
    BatchPredictionResponse,
    BatchRejectedRow,
    CaseDetail,
    CaseSummary,
    PredictionHistoryItem,
    PredictionResponse,
    ReviewRequest,
    ReviewResponse,
    TransactionInput,
)
from fraud_sentinel.settings import Settings, get_settings

PREDICTIONS = Counter("fraud_predictions_total", "Predictions by risk band", ["risk_band"])
CASES = Counter("fraud_cases_created_total", "Cases created by risk band", ["risk_band"])
MODEL_READY = Gauge("fraud_model_ready", "Whether model artifacts are loaded")
RISK_BANDS = {"low", "uncertain", "high"}
CSV_DEMO_ROWS_PER_BAND = 3

FRAUD_LIKE_SAMPLE: dict[str, float] = {
    "Time": 406,
    "Amount": 529,
    "V1": -2.31,
    "V2": 1.95,
    "V3": -1.61,
    "V4": 3.99,
    "V5": -0.52,
    "V6": -1.43,
    "V7": -2.54,
    "V8": 1.39,
    "V9": -2.77,
    "V10": -2.77,
    "V11": 3.2,
    "V12": -2.9,
    "V13": -0.59,
    "V14": -4.29,
    "V15": 0.39,
    "V16": -1.14,
    "V17": -2.83,
    "V18": -0.02,
    "V19": 0.42,
    "V20": 0.13,
    "V21": 0.52,
    "V22": -0.04,
    "V23": -0.47,
    "V24": 0.32,
    "V25": 0.04,
    "V26": 0.18,
    "V27": 0.26,
    "V28": -0.14,
}


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


def _zero_transaction(*, amount: float, time: float) -> dict[str, float]:
    transaction = {column: 0.0 for column in FEATURE_COLUMNS}
    transaction["Time"] = time
    transaction["Amount"] = amount
    return transaction


def _interpolate_transaction(
    left: dict[str, float],
    right: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    return {
        column: float(left[column] * (1.0 - alpha) + right[column] * alpha)
        for column in FEATURE_COLUMNS
    }


def _candidate_demo_transactions() -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [
        _zero_transaction(amount=43.75, time=0),
        dict(FRAUD_LIKE_SAMPLE),
    ]
    for index, amount in enumerate((10, 25, 50, 100, 250, 750, 1500, 2500, 5000)):
        candidates.append(_zero_transaction(amount=float(amount), time=float(index * 300)))

    baseline = _zero_transaction(amount=43.75, time=0)
    for step in range(0, 51):
        candidates.append(_interpolate_transaction(baseline, FRAUD_LIKE_SAMPLE, step / 50))

    index = 0
    for amount in (10, 25, 50, 100, 180, 300, 500, 800, 1200, 2400):
        for v14 in (0, -0.2, -0.5, -1, -1.5, -2, -2.5, -3, -4, 1, 2, 3):
            row = _zero_transaction(amount=float(amount), time=float(7200 + index * 60))
            row["V10"] = float(v14 / 3)
            row["V14"] = float(v14)
            row["V17"] = float(v14 / 2)
            candidates.append(row)
            index += 1
    return candidates


def _spread_rows(rows: list[tuple[dict[str, float], Any]], count: int) -> list[tuple[dict[str, float], Any]]:
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def _demo_csv(model) -> str:
    grouped: dict[str, list[tuple[dict[str, float], Any]]] = {band: [] for band in RISK_BANDS}
    for transaction in _candidate_demo_transactions():
        prediction = model.predict(transaction)
        grouped[prediction.risk_band].append((transaction, prediction))

    grouped["low"].sort(key=lambda item: (item[1].risk_score, item[1].anomaly_score))
    grouped["uncertain"].sort(key=lambda item: (item[1].risk_score, item[1].anomaly_score))
    grouped["high"].sort(key=lambda item: (item[1].risk_score, item[1].anomaly_score))

    selected = [
        *_spread_rows(grouped["low"], CSV_DEMO_ROWS_PER_BAND),
        *_spread_rows(grouped["uncertain"], CSV_DEMO_ROWS_PER_BAND),
        *_spread_rows(grouped["high"], CSV_DEMO_ROWS_PER_BAND),
    ]

    buffer = StringIO()
    columns = [*FEATURE_COLUMNS, LABEL_COLUMN]
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for transaction, prediction in selected:
        writer.writerow(
            {
                **{column: transaction[column] for column in FEATURE_COLUMNS},
                LABEL_COLUMN: 1 if prediction.risk_band == "high" else 0,
            }
        )
    return buffer.getvalue()


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
    rows: list[BatchPredictionRow] = []
    rejections: list[BatchRejectedRow] = []
    rejected = 0
    for row_index, record in enumerate(df.to_dict(orient="records"), start=1):
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
        except Exception as exc:
            rejected += 1
            rejections.append(BatchRejectedRow(row_index=row_index, reason=str(exc)))
            continue
        prediction_ids.append(prediction_id)
        if case_id:
            case_ids.append(case_id)
            CASES.labels(risk_band=model_prediction.risk_band).inc()
        rows.append(
            BatchPredictionRow(
                row_index=row_index,
                prediction_id=prediction_id,
                case_id=case_id,
                **prediction_data,
            )
        )
        PREDICTIONS.labels(risk_band=model_prediction.risk_band).inc()

    return BatchPredictionResponse(
        accepted_rows=len(prediction_ids),
        rejected_rows=rejected,
        prediction_ids=prediction_ids,
        case_ids=case_ids,
        rows=rows,
        rejections=rejections,
    )


@app.get("/v1/predictions", response_model=list[PredictionHistoryItem])
async def list_predictions(
    risk_band: str | None = None,
    has_case: bool | None = None,
    limit: int = 200,
    repo: Repository = Depends(get_repo),
) -> list[dict[str, Any]]:
    if risk_band is not None and risk_band not in RISK_BANDS:
        raise HTTPException(status_code=400, detail="risk_band must be one of low, uncertain, high")
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return await repo.list_predictions(risk_band=risk_band, has_case=has_case, limit=limit)


@app.get("/v1/samples/demo.csv")
async def sample_demo_csv(model=Depends(get_model)) -> Response:
    return Response(
        content=_demo_csv(model),
        media_type="text/csv",
        headers={"content-disposition": 'attachment; filename="fraud-sentinel-demo.csv"'},
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
