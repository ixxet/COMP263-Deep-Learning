"""Training pipeline and model gates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_sentinel.feature_schema import FEATURE_COLUMNS, LABEL_COLUMN, check_columns
from fraud_sentinel.model.network import DenseFraudClassifier, FraudAutoencoder
from fraud_sentinel.risk import Thresholds


@dataclass(frozen=True)
class TrainingResult:
    model_version: str
    metrics: dict
    thresholds: Thresholds
    output_dir: str


def load_creditcard_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    check = check_columns(df.columns, include_label=True)
    if not check.ok:
        raise ValueError(check.message())
    if df[LABEL_COLUMN].isna().any():
        raise ValueError("Class label contains missing values")
    return df.loc[:, list(FEATURE_COLUMNS) + [LABEL_COLUMN]]


def train(
    csv_path: Path,
    output_dir: Path,
    *,
    epochs: int = 12,
    batch_size: int = 2048,
    min_pr_auc: float = 0.70,
    min_recall: float = 0.80,
    seed: int = 263,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_creditcard_csv(csv_path)
    y = df[LABEL_COLUMN].astype(int).to_numpy()
    x = df.loc[:, FEATURE_COLUMNS].astype("float32").to_numpy()
    class_counts = {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=seed, stratify=y_train
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype("float32")
    x_val_s = scaler.transform(x_val).astype("float32")
    x_test_s = scaler.transform(x_test).astype("float32")

    classifier = DenseFraudClassifier(input_dim=x_train_s.shape[1]).to(device)
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)],
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y_train.astype("float32"))),
        batch_size=batch_size,
        shuffle=True,
    )
    classifier.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(classifier(xb), yb)
            loss.backward()
            optimizer.step()

    autoencoder = FraudAutoencoder(input_dim=x_train_s.shape[1]).to(device)
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    normal_train = x_train_s[y_train == 0]
    ae_loader = DataLoader(TensorDataset(torch.from_numpy(normal_train)), batch_size=batch_size)
    mse = nn.MSELoss()
    autoencoder.train()
    for _ in range(max(6, epochs // 2)):
        for (xb,) in ae_loader:
            xb = xb.to(device)
            ae_optimizer.zero_grad()
            reconstructed = autoencoder(xb)
            loss = mse(reconstructed, xb)
            loss.backward()
            ae_optimizer.step()

    val_scores = _predict_classifier(classifier, x_val_s)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, val_scores)
    validation_target_recall = min(0.98, min_recall + 0.05)
    selected_threshold = _select_threshold(
        precision_curve,
        recall_curve,
        thresholds,
        target_recall=validation_target_recall,
    )

    test_scores = _predict_classifier(classifier, x_test_s)
    test_pred = (test_scores >= selected_threshold).astype(int)
    test_errors = _reconstruction_errors(autoencoder, x_test_s)
    anomaly_min = float(np.percentile(test_errors, 1))
    anomaly_max = float(np.percentile(test_errors, 99))
    anomaly_threshold = float(np.percentile(test_errors[y_test == 0], 99.5))
    anomaly_norm_threshold = max(
        0.0, min(1.0, (anomaly_threshold - anomaly_min) / max(anomaly_max - anomaly_min, 1e-9))
    )

    metrics = {
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "selected_risk_threshold": float(selected_threshold),
        "validation_target_recall": float(validation_target_recall),
        "class_counts": class_counts,
        "training_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
    }

    if metrics["average_precision"] < min_pr_auc or metrics["recall"] < min_recall:
        raise RuntimeError(
            "G2 model gate failed: "
            f"PR-AUC={metrics['average_precision']:.3f}, recall={metrics['recall']:.3f}"
        )

    model_version = f"fraud-sentinel-{int(time.time())}"
    thresholds_obj = Thresholds(
        low_risk_score=max(0.05, selected_threshold * 0.65),
        high_risk_score=float(selected_threshold),
        elevated_anomaly_score=max(0.50, anomaly_norm_threshold * 0.75),
        high_anomaly_score=max(0.70, anomaly_norm_threshold),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), output_dir / "classifier.pt")
    torch.save(autoencoder.state_dict(), output_dir / "autoencoder.pt")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    metadata = {
        "model_version": model_version,
        "feature_columns": list(FEATURE_COLUMNS),
        "thresholds": asdict(thresholds_obj),
        "anomaly_stats": {"min": anomaly_min, "max": anomaly_max},
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(
        model_version=model_version,
        metrics=metrics,
        thresholds=thresholds_obj,
        output_dir=str(output_dir),
    )


def _predict_classifier(model: DenseFraudClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def _reconstruction_errors(model: FraudAutoencoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(device)
        reconstructed = model(tensor)
        return torch.mean((tensor - reconstructed) ** 2, dim=1).cpu().numpy()


def _select_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    *,
    target_recall: float = 0.80,
) -> float:
    if len(thresholds) == 0:
        return 0.5
    candidates = []
    for p, r, threshold in zip(precision[:-1], recall[:-1], thresholds):
        if r >= target_recall:
            f1 = 2 * p * r / max(p + r, 1e-9)
            candidates.append((f1, threshold))
    if candidates:
        return float(max(candidates, key=lambda item: item[0])[1])
    best_idx = int(np.argmax(2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)))
    return float(thresholds[best_idx])
