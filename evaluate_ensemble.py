#!/usr/bin/env python
"""Evaluate ensemble of deep learning models."""

from pathlib import Path

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models.deep import DeepEnsemble


def main():
    # Load data
    print("Loading data...")
    df = load_ohlcv("data/BTC.parquet")
    print(f"Loaded {len(df)} rows")

    # Extract features
    print("\nExtracting features...")
    extractor = FeatureExtractor(include_advanced=True)
    features = extractor.transform(df)
    print(f"Features: {len(features.columns)} columns")

    # Generate labels
    print("\nGenerating labels...")
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Use last 20% as test set
    n_test = int(len(features) * 0.2)
    test_features = features.iloc[-n_test:]
    test_labels = labels.iloc[-n_test:]
    print(f"Test samples: {len(test_features)}")

    # Load ensemble
    print("\n" + "=" * 60)
    print("Loading Ensemble")
    print("=" * 60)

    model_dir = Path("models")
    ensemble = DeepEnsemble.from_saved_models(
        lstm_path=model_dir / "regime_classifier_lstm_v2.pt",
        transformer_path=model_dir / "regime_classifier_transformer_v2.pt",
        cnn_lstm_path=model_dir / "regime_classifier_cnn_lstm_v2.pt",
        voting="soft",
    )

    # Compare individual models vs ensemble
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    comparison = ensemble.compare_models(test_features, test_labels)
    print(comparison.to_string(index=False))

    # Detailed ensemble evaluation
    print("\n" + "=" * 60)
    print("Ensemble Detailed Evaluation")
    print("=" * 60)
    results = ensemble.evaluate(test_features, test_labels)

    # Try different thresholds for 2-class
    print("\n" + "=" * 60)
    print("Threshold Analysis (for BULL_TREND)")
    print("=" * 60)

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Get probabilities
    proba = ensemble.predict_proba(test_features)
    common_idx = proba.index.intersection(test_labels.index)
    y_true = test_labels.loc[common_idx]
    bull_proba = proba.loc[common_idx, "BULL_TREND"]

    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = ["BULL_TREND" if p >= threshold else "NOT_BULL" for p in bull_proba]

        precision = precision_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="BULL_TREND", zero_division=0)

        print(f"{threshold:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")


if __name__ == "__main__":
    main()
