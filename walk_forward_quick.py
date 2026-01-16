#!/usr/bin/env python
"""Quick walk-forward validation for regime classifier."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import LSTMClassifier


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")
    print(f"Loaded {len(df)} rows")

    # Extract features
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)

    # Generate labels
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Align
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Model parameters - reduced for speed
    model_params = {
        "seq_length": 30,     # Reduced from 60
        "hidden_size": 64,    # Reduced from 128
        "num_layers": 1,      # Reduced from 2
        "n_classes": 2,
        "learning_rate": 0.001,
        "batch_size": 64,     # Increased for speed
        "epochs": 30,         # Reduced from 50
        "patience": 5,        # Reduced from 10
        "random_state": 42,
    }

    # Walk-forward parameters
    train_size = 400   # ~1.3 years
    test_size = 150    # ~5 months
    step_size = 150    # Move forward same as test_size
    seq_length = model_params["seq_length"]

    n_samples = len(features)
    results = []
    fold = 0
    start_idx = 0

    print(f"\nTotal samples: {n_samples}")
    print(f"Train size: {train_size}, Test size: {test_size}, Step: {step_size}")
    print(f"Expected folds: ~{(n_samples - train_size - test_size) // step_size + 1}")

    while start_idx + train_size + test_size <= n_samples:
        fold += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Split data
        train_features = features.iloc[start_idx:train_end]
        train_labels = labels.iloc[start_idx:train_end]
        test_features = features.iloc[train_end:test_end]
        test_labels = labels.iloc[train_end:test_end]

        print(f"\n{'='*60}")
        print(f"Fold {fold}: Train {train_features.index.min().date()} to {train_features.index.max().date()}")
        print(f"         Test  {test_features.index.min().date()} to {test_features.index.max().date()}")

        # Train model
        model = LSTMClassifier(**model_params)
        model.fit(train_features, train_labels, eval_split=0.15, verbose=False)

        # Predict
        predictions = model.predict(test_features)
        if isinstance(predictions, str):
            predictions = pd.Series([predictions], index=[test_features.index[-1]])

        # Align with labels
        valid_test_labels = test_labels.iloc[seq_length - 1:]
        common_idx = predictions.index.intersection(valid_test_labels.index)

        if len(common_idx) == 0:
            print(f"  No valid samples!")
            start_idx += step_size
            continue

        y_true = valid_test_labels.loc[common_idx]
        y_pred = predictions.loc[common_idx]

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)

        n_bull_actual = (y_true == "BULL_TREND").sum()
        n_bull_pred = (y_pred == "BULL_TREND").sum()

        results.append({
            "fold": fold,
            "test_start": test_features.index.min(),
            "test_end": test_features.index.max(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_samples": len(y_true),
            "n_bull_actual": n_bull_actual,
            "n_bull_pred": n_bull_pred,
        })

        print(f"  Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
        print(f"  BULL actual: {n_bull_actual}, pred: {n_bull_pred}")

        start_idx += step_size

    # Summary
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total folds: {len(df_results)}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy:  {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
    print(f"  Precision: {df_results['precision'].mean():.4f} ± {df_results['precision'].std():.4f}")
    print(f"  Recall:    {df_results['recall'].mean():.4f} ± {df_results['recall'].std():.4f}")
    print(f"  F1:        {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")

    print("\nPer-fold results:")
    for _, row in df_results.iterrows():
        print(f"  Fold {row['fold']}: {row['test_start'].date()} ~ {row['test_end'].date()} | "
              f"Acc={row['accuracy']:.3f} Prec={row['precision']:.3f} Rec={row['recall']:.3f}")

    df_results.to_csv("models/walk_forward_quick_results.csv", index=False)
    print(f"\nResults saved to models/walk_forward_quick_results.csv")


if __name__ == "__main__":
    main()
