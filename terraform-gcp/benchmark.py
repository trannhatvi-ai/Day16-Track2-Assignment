#!/usr/bin/env python3
"""LightGBM benchmark for Part 7 CPU fallback path on GCP.

Expected dataset: creditcard.csv from Kaggle (mlg-ulb/creditcardfraud).
Run from VM:
  python3 benchmark.py

Optional args:
  python3 benchmark.py --data /path/to/creditcard.csv --output benchmark_result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGBM benchmark on credit card fraud dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard.csv",
        help="Path to creditcard.csv (default: creditcard.csv in current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_result.json",
        help="Path to output JSON file (default: benchmark_result.json)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=100,
        help="Number of runs to estimate single-row latency (default: 100)",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = {"Class"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def measure_single_row_latency(model: Any, row: np.ndarray, runs: int) -> float:
    # Median of multiple runs gives a more stable latency estimate.
    durations_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.predict_proba(row)
        end = time.perf_counter()
        durations_ms.append((end - start) * 1000.0)
    return float(statistics.median(durations_ms))


def measure_throughput(model: Any, batch: np.ndarray) -> float:
    start = time.perf_counter()
    _ = model.predict_proba(batch)
    end = time.perf_counter()
    elapsed = end - start
    if elapsed <= 0:
        return float("inf")
    return float(len(batch) / elapsed)


def main() -> None:
    args = parse_args()

    try:
        lgb = importlib.import_module("lightgbm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'lightgbm'. Install with: pip3 install lightgbm"
        ) from exc

    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download with: "
            "kaggle datasets download -d mlg-ulb/creditcardfraud --unzip"
        )

    print(f"[1/5] Loading dataset from: {data_path}")
    data_load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    data_load_end = time.perf_counter()
    validate_dataset(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("[2/5] Splitting train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    print("[3/5] Training LightGBM model")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        n_jobs=os.cpu_count() or 4,
    )

    training_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
        ],
    )
    training_end = time.perf_counter()

    print("[4/5] Evaluating metrics")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))

    best_iteration = int(model.best_iteration_ or model.n_estimators)

    print("[5/5] Measuring inference performance")
    first_row = X_test.iloc[[0]].to_numpy()
    batch_1000 = X_test.iloc[: min(1000, len(X_test))].to_numpy()

    inference_latency_ms = measure_single_row_latency(model, first_row, runs=max(1, args.latency_runs))
    inference_throughput = measure_throughput(model, batch_1000)

    results: dict[str, Any] = {
        "dataset": str(data_path),
        "data_rows": int(len(df)),
        "data_columns": int(len(df.columns)),
        "test_size": float(args.test_size),
        "random_seed": int(args.seed),
        "load_data_time_sec": float(data_load_end - data_load_start),
        "training_time_sec": float(training_end - training_start),
        "best_iteration": best_iteration,
        "auc_roc": auc_roc,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "inference_latency_ms": inference_latency_ms,
        "inference_throughput_rows_per_sec": inference_throughput,
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n===== Benchmark Summary =====")
    print(f"Load data time (sec): {results['load_data_time_sec']:.4f}")
    print(f"Training time (sec):  {results['training_time_sec']:.4f}")
    print(f"Best iteration:       {results['best_iteration']}")
    print(f"AUC-ROC:              {results['auc_roc']:.6f}")
    print(f"Accuracy:             {results['accuracy']:.6f}")
    print(f"F1-score:             {results['f1_score']:.6f}")
    print(f"Precision:            {results['precision']:.6f}")
    print(f"Recall:               {results['recall']:.6f}")
    print(f"Inference latency ms: {results['inference_latency_ms']:.6f}")
    print(f"Inference throughput: {results['inference_throughput_rows_per_sec']:.2f} rows/sec")
    print(f"Saved JSON result to: {output_path}")


if __name__ == "__main__":
    main()
