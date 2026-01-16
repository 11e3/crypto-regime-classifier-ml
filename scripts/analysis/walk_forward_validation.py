#!/usr/bin/env python
"""Walk-forward validation for regime classifier."""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import LSTMClassifier, TransformerClassifier, CNNLSTMClassifier


def walk_forward_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    model_class,
    model_params: dict,
    train_size: int = 500,
    test_size: int = 60,
    step_size: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform walk-forward validation.

    Args:
        features: Feature DataFrame
        labels: Labels Series
        model_class: Model class to use
        model_params: Parameters for the model
        train_size: Initial training window size
        test_size: Test window size
        step_size: How many steps to move forward each iteration
        verbose: Whether to print progress

    Returns:
        DataFrame with results for each fold
    """
    results = []
    n_samples = len(features)

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    fold = 0
    start_idx = 0

    while start_idx + train_size + test_size <= n_samples:
        fold += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Split data
        train_features = features.iloc[start_idx:train_end]
        train_labels = labels.iloc[start_idx:train_end]
        test_features = features.iloc[train_end:test_end]
        test_labels = labels.iloc[train_end:test_end]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"Train: {train_features.index.min().date()} to {train_features.index.max().date()} ({len(train_features)} samples)")
            print(f"Test:  {test_features.index.min().date()} to {test_features.index.max().date()} ({len(test_features)} samples)")

        # Train model
        model = model_class(**model_params)
        model.fit(train_features, train_labels, eval_split=0.15, verbose=False)

        # Predict
        predictions = model.predict(test_features)

        # Handle single prediction case
        if isinstance(predictions, str):
            predictions = pd.Series([predictions], index=[test_features.index[-1]])

        # Align predictions with labels (account for seq_length)
        seq_length = model_params.get("seq_length", 60)
        valid_test_labels = test_labels.iloc[seq_length - 1:]

        if len(predictions) == 0 or len(valid_test_labels) == 0:
            print(f"  Skipping fold {fold} - insufficient samples")
            start_idx += step_size
            continue

        common_idx = predictions.index.intersection(valid_test_labels.index)
        y_true = valid_test_labels.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)

        # Count predictions
        n_bull_pred = (y_pred == "BULL_TREND").sum()
        n_bull_actual = (y_true == "BULL_TREND").sum()

        results.append({
            "fold": fold,
            "train_start": train_features.index.min(),
            "train_end": train_features.index.max(),
            "test_start": test_features.index.min(),
            "test_end": test_features.index.max(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_test_samples": len(y_true),
            "n_bull_actual": n_bull_actual,
            "n_bull_pred": n_bull_pred,
        })

        if verbose:
            print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            print(f"  BULL actual: {n_bull_actual}, BULL predicted: {n_bull_pred}")

        # Move forward
        start_idx += step_size

    return pd.DataFrame(results)


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")
    print(f"Loaded {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")

    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)
    print(f"Features: {len(features.columns)} columns, {len(features)} rows")

    # Generate labels
    print("\nGenerating labels...")
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)
    print(f"Label distribution:")
    print(labels.value_counts())

    # Model parameters
    model_params = {
        "seq_length": 60,
        "hidden_size": 128,
        "num_layers": 2,
        "n_classes": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,  # Reduced for faster validation
        "patience": 10,
        "random_state": 42,
    }

    # Walk-forward validation for LSTM
    print("\n" + "=" * 60)
    print("LSTM Walk-Forward Validation")
    print("=" * 60)

    lstm_results = walk_forward_validation(
        features=features,
        labels=labels,
        model_class=LSTMClassifier,
        model_params=model_params,
        train_size=500,  # ~1.5 years
        test_size=120,   # ~4 months (need > seq_length for valid predictions)
        step_size=90,    # Move 3 months forward each fold
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("LSTM Summary")
    print("=" * 60)
    print(f"Number of folds: {len(lstm_results)}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy:  {lstm_results['accuracy'].mean():.4f} ± {lstm_results['accuracy'].std():.4f}")
    print(f"  Precision: {lstm_results['precision'].mean():.4f} ± {lstm_results['precision'].std():.4f}")
    print(f"  Recall:    {lstm_results['recall'].mean():.4f} ± {lstm_results['recall'].std():.4f}")
    print(f"  F1:        {lstm_results['f1'].mean():.4f} ± {lstm_results['f1'].std():.4f}")

    # Per-fold results
    print("\nPer-fold results:")
    print(lstm_results[["fold", "test_start", "accuracy", "precision", "recall", "f1"]].to_string(index=False))

    # Save results
    lstm_results.to_csv("models/walk_forward_lstm_results.csv", index=False)
    print(f"\nResults saved to models/walk_forward_lstm_results.csv")

    # Walk-forward for Transformer
    print("\n" + "=" * 60)
    print("Transformer Walk-Forward Validation")
    print("=" * 60)

    transformer_results = walk_forward_validation(
        features=features,
        labels=labels,
        model_class=TransformerClassifier,
        model_params=model_params,
        train_size=500,
        test_size=120,
        step_size=90,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Transformer Summary")
    print("=" * 60)
    print(f"Number of folds: {len(transformer_results)}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy:  {transformer_results['accuracy'].mean():.4f} ± {transformer_results['accuracy'].std():.4f}")
    print(f"  Precision: {transformer_results['precision'].mean():.4f} ± {transformer_results['precision'].std():.4f}")
    print(f"  Recall:    {transformer_results['recall'].mean():.4f} ± {transformer_results['recall'].std():.4f}")
    print(f"  F1:        {transformer_results['f1'].mean():.4f} ± {transformer_results['f1'].std():.4f}")

    transformer_results.to_csv("models/walk_forward_transformer_results.csv", index=False)

    # Walk-forward for CNN-LSTM
    print("\n" + "=" * 60)
    print("CNN-LSTM Walk-Forward Validation")
    print("=" * 60)

    cnn_lstm_results = walk_forward_validation(
        features=features,
        labels=labels,
        model_class=CNNLSTMClassifier,
        model_params=model_params,
        train_size=500,
        test_size=120,
        step_size=90,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("CNN-LSTM Summary")
    print("=" * 60)
    print(f"Number of folds: {len(cnn_lstm_results)}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy:  {cnn_lstm_results['accuracy'].mean():.4f} ± {cnn_lstm_results['accuracy'].std():.4f}")
    print(f"  Precision: {cnn_lstm_results['precision'].mean():.4f} ± {cnn_lstm_results['precision'].std():.4f}")
    print(f"  Recall:    {cnn_lstm_results['recall'].mean():.4f} ± {cnn_lstm_results['recall'].std():.4f}")
    print(f"  F1:        {cnn_lstm_results['f1'].mean():.4f} ± {cnn_lstm_results['f1'].std():.4f}")

    cnn_lstm_results.to_csv("models/walk_forward_cnn_lstm_results.csv", index=False)

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    comparison = pd.DataFrame({
        "Model": ["LSTM", "Transformer", "CNN-LSTM"],
        "Accuracy": [
            f"{lstm_results['accuracy'].mean():.4f} ± {lstm_results['accuracy'].std():.4f}",
            f"{transformer_results['accuracy'].mean():.4f} ± {transformer_results['accuracy'].std():.4f}",
            f"{cnn_lstm_results['accuracy'].mean():.4f} ± {cnn_lstm_results['accuracy'].std():.4f}",
        ],
        "Precision": [
            f"{lstm_results['precision'].mean():.4f} ± {lstm_results['precision'].std():.4f}",
            f"{transformer_results['precision'].mean():.4f} ± {transformer_results['precision'].std():.4f}",
            f"{cnn_lstm_results['precision'].mean():.4f} ± {cnn_lstm_results['precision'].std():.4f}",
        ],
        "Recall": [
            f"{lstm_results['recall'].mean():.4f} ± {lstm_results['recall'].std():.4f}",
            f"{transformer_results['recall'].mean():.4f} ± {transformer_results['recall'].std():.4f}",
            f"{cnn_lstm_results['recall'].mean():.4f} ± {cnn_lstm_results['recall'].std():.4f}",
        ],
        "F1": [
            f"{lstm_results['f1'].mean():.4f} ± {lstm_results['f1'].std():.4f}",
            f"{transformer_results['f1'].mean():.4f} ± {transformer_results['f1'].std():.4f}",
            f"{cnn_lstm_results['f1'].mean():.4f} ± {cnn_lstm_results['f1'].std():.4f}",
        ],
    })
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
