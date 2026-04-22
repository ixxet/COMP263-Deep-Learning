"""Model artifact loading and deterministic prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import torch

from fraud_sentinel.feature_schema import FEATURE_COLUMNS, ordered_feature_vector
from fraud_sentinel.model.network import DenseFraudClassifier, FraudAutoencoder
from fraud_sentinel.risk import Thresholds, risk_band


@dataclass(frozen=True)
class Prediction:
    risk_score: float
    anomaly_score: float
    risk_band: str
    model_version: str


class ModelBundle:
    def __init__(
        self,
        classifier: DenseFraudClassifier,
        autoencoder: FraudAutoencoder,
        scaler,
        thresholds: Thresholds,
        model_version: str,
        anomaly_min: float,
        anomaly_max: float,
    ) -> None:
        self.classifier = classifier.eval()
        self.autoencoder = autoencoder.eval()
        self.scaler = scaler
        self.thresholds = thresholds
        self.model_version = model_version
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max

    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        metadata_path = model_dir / "metadata.json"
        classifier_path = model_dir / "classifier.pt"
        autoencoder_path = model_dir / "autoencoder.pt"
        scaler_path = model_dir / "scaler.joblib"

        missing = [
            str(path)
            for path in (metadata_path, classifier_path, autoencoder_path, scaler_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"model artifact bundle incomplete: {', '.join(missing)}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classifier = DenseFraudClassifier(input_dim=len(FEATURE_COLUMNS))
        autoencoder = FraudAutoencoder(input_dim=len(FEATURE_COLUMNS))
        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location="cpu"))
        scaler = joblib.load(scaler_path)

        anomaly_stats = metadata.get("anomaly_stats", {})
        return cls(
            classifier=classifier,
            autoencoder=autoencoder,
            scaler=scaler,
            thresholds=Thresholds.from_mapping(metadata.get("thresholds")),
            model_version=str(metadata.get("model_version", "unknown")),
            anomaly_min=float(anomaly_stats.get("min", 0.0)),
            anomaly_max=float(anomaly_stats.get("max", 1.0)),
        )

    def predict(self, transaction: Mapping[str, float]) -> Prediction:
        vector = np.asarray([ordered_feature_vector(transaction)], dtype=np.float32)
        scaled = self.scaler.transform(vector).astype(np.float32)
        tensor = torch.from_numpy(scaled)
        with torch.no_grad():
            risk = torch.sigmoid(self.classifier(tensor)).item()
            reconstructed = self.autoencoder(tensor)
            reconstruction_error = torch.mean((tensor - reconstructed) ** 2, dim=1).item()
        anomaly = self._normalize_anomaly(reconstruction_error)
        band = risk_band(risk, anomaly, self.thresholds)
        return Prediction(
            risk_score=float(risk),
            anomaly_score=float(anomaly),
            risk_band=band,
            model_version=self.model_version,
        )

    def _normalize_anomaly(self, reconstruction_error: float) -> float:
        span = max(self.anomaly_max - self.anomaly_min, 1e-9)
        return max(0.0, min(1.0, (reconstruction_error - self.anomaly_min) / span))


class DemoModelBundle:
    """Explicit local-only model for UI/API smoke testing before training artifacts exist."""

    model_version = "demo-untrained"

    def predict(self, transaction: Mapping[str, float]) -> Prediction:
        amount = max(float(transaction["Amount"]), 0.0)
        velocity = abs(float(transaction["V14"])) + abs(float(transaction["V17"]))
        risk = min(0.99, 0.05 + amount / 5000.0 + velocity / 12.0)
        anomaly = min(0.99, amount / 10000.0 + abs(float(transaction["V10"])) / 10.0)
        thresholds = Thresholds()
        return Prediction(
            risk_score=risk,
            anomaly_score=anomaly,
            risk_band=risk_band(risk, anomaly, thresholds),
            model_version=self.model_version,
        )

