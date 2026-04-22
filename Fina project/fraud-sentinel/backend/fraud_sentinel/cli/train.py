"""Train Fraud Sentinel model artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fraud_sentinel.model.training import train

DATASET = "mlg-ulb/creditcardfraud"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fraud Sentinel models.")
    parser.add_argument("--csv", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/models"))
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-pr-auc", type=float, default=0.70)
    parser.add_argument("--min-recall", type=float, default=0.80)
    args = parser.parse_args()

    if args.download_if_missing and not args.csv.exists():
        download_kaggle_dataset(args.csv.parent)

    result = train(
        args.csv,
        args.output_dir,
        epochs=args.epochs,
        min_pr_auc=args.min_pr_auc,
        min_recall=args.min_recall,
    )
    print(json.dumps({"model_version": result.model_version, "metrics": result.metrics}, indent=2))


def download_kaggle_dataset(output_dir: Path) -> None:
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY are required for Kaggle download")
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(output_dir), unzip=True)


if __name__ == "__main__":
    main()

