#!/usr/bin/env python
"""Evaluate LSTM model using walk-forward validation.

This script evaluates the LSTM model with proper walk-forward validation
to compare with RF and XGBoost results.

Usage:
    python evaluate_lstm_walk_forward.py
    python evaluate_lstm_walk_forward.py --epochs 10  # faster training per fold
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import LSTMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Walk-Forward Validation")
    parser.add_argument("--data", type=str, default="data/BTC.parquet", help="Path to OHLCV data")
    parser.add_argument("--n-classes", type=int, default=2, help="Number of regime classes")
    parser.add_argument("--train-size", type=int, default=500, help="Training window size")
    parser.add_argument("--test-size", type=int, default=60, help="Test window size")
    parser.add_argument("--step-size", type=int, default=60, help="Step size for validation")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per fold")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def lstm_walk_forward_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    args,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform walk-forward validation for LSTM model."""
    results = []
    n_samples = len(features)

    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    fold = 0
    start_idx = 0
    seq_length = args.seq_length

    while start_idx + args.train_size + args.test_size <= n_samples:
        fold += 1
        train_end = start_idx + args.train_size
        test_end = train_end + args.test_size

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

        # Create and train LSTM
        model = LSTMClassifier(
            seq_length=seq_length,
            hidden_size=args.hidden_size,
            num_layers=2,
            n_classes=args.n_classes,
            dropout=0.3,
            use_attention=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            patience=5,
        )

        # Train model
        model.fit(
            train_features,
            train_labels,
            eval_split=0.15,
            verbose=False,
        )

        # Predict - LSTM needs seq_length samples to make prediction
        # For test set, we need to include seq_length-1 samples from training for context
        context_start = train_end - seq_length + 1
        full_test_features = features.iloc[context_start:test_end]

        predictions = model.predict(full_test_features)

        if isinstance(predictions, str):
            predictions = pd.Series([predictions], index=[full_test_features.index[-1]])

        # Test labels (only the actual test period)
        valid_test_labels = test_labels

        if len(predictions) == 0 or len(valid_test_labels) == 0:
            print(f"  Skipping fold {fold} - insufficient samples")
            start_idx += args.step_size
            continue

        # Align predictions with labels
        common_idx = predictions.index.intersection(valid_test_labels.index)
        if len(common_idx) == 0:
            print(f"  Skipping fold {fold} - no matching indices")
            start_idx += args.step_size
            continue

        y_true = valid_test_labels.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)

        results.append({
            "fold": fold,
            "train_start": train_features.index.min(),
            "train_end": train_features.index.max(),
            "test_start": test_features.index.min(),
            "test_end": test_features.index.max(),
            "train_samples": len(train_features),
            "test_samples": len(y_true),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        if verbose:
            print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Move forward
        start_idx += args.step_size

    return pd.DataFrame(results)


def main():
    args = parse_args()

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")

    # Extract features
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)
    print(f"Features: {len(features.columns)} columns, {len(features)} rows")

    # Generate labels
    print("\n" + "=" * 70)
    print("GENERATING LABELS")
    print("=" * 70)
    labeler = RegimeLabeler(n_classes=args.n_classes)
    labels = labeler.label(df)
    print(f"Label distribution:")
    print(labels.value_counts())

    # LSTM Walk-Forward Validation
    print("\n" + "=" * 70)
    print("LSTM WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Config: train={args.train_size}, test={args.test_size}, step={args.step_size}")
    print(f"LSTM: seq_length={args.seq_length}, hidden={args.hidden_size}, epochs={args.epochs}")

    results_df = lstm_walk_forward_validation(features, labels, args, verbose=True)

    # Summary
    print("\n" + "=" * 70)
    print("LSTM WALK-FORWARD SUMMARY")
    print("=" * 70)

    if len(results_df) > 0:
        print(f"Number of folds: {len(results_df)}")
        print(f"Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
        print(f"Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
        print(f"F1:        {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")

        # Comparison with supervised models
        print("\n" + "=" * 70)
        print("COMPARISON WITH SUPERVISED MODELS")
        print("=" * 70)
        print(f"{'Model':<15} {'Accuracy':<20} {'F1':<20}")
        print("-" * 55)
        print(f"{'Random Forest':<15} {'0.8734 +/- 0.1381':<20} {'0.7263 +/- 0.3249':<20}")
        print(f"{'XGBoost':<15} {'0.8883 +/- 0.1278':<20} {'0.7505 +/- 0.3113':<20}")
        lstm_acc = f"{results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}"
        lstm_f1 = f"{results_df['f1'].mean():.4f} +/- {results_df['f1'].std():.4f}"
        print(f"{'LSTM':<15} {lstm_acc:<20} {lstm_f1:<20}")
    else:
        print("No results generated")

    # Save results
    results_df.to_csv("lstm_walk_forward_results.csv", index=False)
    print(f"\nResults saved to lstm_walk_forward_results.csv")


if __name__ == "__main__":
    main()
