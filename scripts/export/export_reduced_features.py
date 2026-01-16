#!/usr/bin/env python
"""Export XGBoost model with reduced features (18 features).

Usage:
    python export_reduced_features.py
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier
from src.validation.walk_forward import walk_forward_validation


# Curated 18 features based on PCA analysis
REDUCED_FEATURES = [
    # Returns (3)
    "return_1d", "return_5d", "return_20d",
    # Volatility (2)
    "volatility", "atr_pct",
    # Momentum (3)
    "rsi", "momentum_20", "macd_histogram",
    # Volume (3)
    "volume_ratio_20", "obv_trend", "volume_price_corr",
    # Trend (4)
    "ma_alignment", "ma_20_slope", "trend_strength", "pivot_trend_score",
    # Bollinger (1)
    "bb_position",
    # Range (2)
    "range_position", "breakout_potential",
]


def main():
    output_dir = Path("../crypto-quant-system/models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export XGBoost with Reduced Features (18)")
    print("=" * 60)
    print(f"Output: {output_dir}")

    # Load data
    print("\n[1/5] Loading data...")
    df = load_ohlcv("data/BTC.parquet")
    print(f"  Loaded {len(df)} rows")

    # Extract all features
    print("\n[2/5] Extracting features...")
    extractor = FeatureExtractor(include_advanced=False)
    features_full = extractor.transform(df)

    # Select reduced features
    features = features_full[REDUCED_FEATURES].copy()
    print(f"  Reduced: {len(features_full.columns)} -> {len(features.columns)} features")

    # Generate labels
    print("\n[3/5] Generating labels...")
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    print(f"  Samples: {len(X)}")

    # Walk-forward validation to compare performance
    print("\n[4/5] Walk-forward validation (comparing full vs reduced)...")
    print("-" * 40)

    # Full features
    X_full = features_full.loc[common_idx]
    results_full = walk_forward_validation(
        features=X_full,
        labels=y,
        model_class=RegimeClassifier,
        model_params={"model_type": "xgboost", "scale_features": True, "random_state": 42},
        train_size=500,
        test_size=60,
        step_size=60,
        verbose=False,
    )

    # Reduced features
    results_reduced = walk_forward_validation(
        features=X,
        labels=y,
        model_class=RegimeClassifier,
        model_params={"model_type": "xgboost", "scale_features": True, "random_state": 42},
        train_size=500,
        test_size=60,
        step_size=60,
        verbose=False,
    )

    print(f"\n{'Features':<15} {'Accuracy':<20} {'F1':<20}")
    print("-" * 55)
    print(f"{'Full (51)':<15} {results_full['accuracy'].mean():.4f} ± {results_full['accuracy'].std():.4f}    {results_full['f1'].mean():.4f} ± {results_full['f1'].std():.4f}")
    print(f"{'Reduced (18)':<15} {results_reduced['accuracy'].mean():.4f} ± {results_reduced['accuracy'].std():.4f}    {results_reduced['f1'].mean():.4f} ± {results_reduced['f1'].std():.4f}")

    acc_diff = results_reduced['accuracy'].mean() - results_full['accuracy'].mean()
    print(f"\nAccuracy change: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")

    # Train final model with reduced features
    print("\n[5/5] Training final model...")
    model = RegimeClassifier(
        model_type="xgboost",
        scale_features=True,
        random_state=42,
    )
    model.fit(X, y, eval_split=0.2, verbose=True)

    # Export
    print("\n" + "=" * 60)
    print("Exporting...")
    print("=" * 60)

    export_data = {
        # Core model components
        "model": model.model,
        "scaler": model.scaler,
        "label_encoder": model.label_encoder,
        "feature_names": REDUCED_FEATURES,
        "classes": list(model.label_encoder.classes_),

        # Metadata
        "model_type": "xgboost",
        "n_classes": 2,
        "n_features": len(REDUCED_FEATURES),
        "training_samples": len(X),
        "training_date_range": {
            "start": str(X.index.min().date()),
            "end": str(X.index.max().date()),
        },

        # Performance
        "performance": {
            "accuracy_mean": float(results_reduced['accuracy'].mean()),
            "accuracy_std": float(results_reduced['accuracy'].std()),
            "f1_mean": float(results_reduced['f1'].mean()),
            "f1_std": float(results_reduced['f1'].std()),
        },
    }

    output_path = output_dir / "regime_classifier_xgb_reduced.joblib"
    joblib.dump(export_data, output_path)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
Model: XGBoost with {len(REDUCED_FEATURES)} features
File:  regime_classifier_xgb_reduced.joblib

Performance (Walk-Forward):
  Accuracy: {results_reduced['accuracy'].mean():.2%} ± {results_reduced['accuracy'].std():.2%}
  F1 Score: {results_reduced['f1'].mean():.4f} ± {results_reduced['f1'].std():.4f}

Features ({len(REDUCED_FEATURES)}):
{chr(10).join(f'  - {f}' for f in REDUCED_FEATURES)}

Usage:
  clf = joblib.load("models/regime_classifier_xgb_reduced.joblib")
  X_scaled = clf["scaler"].transform(features[clf["feature_names"]])
  pred = clf["label_encoder"].inverse_transform(clf["model"].predict(X_scaled))
""")


if __name__ == "__main__":
    main()
