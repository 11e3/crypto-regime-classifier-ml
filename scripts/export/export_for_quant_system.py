#!/usr/bin/env python
"""Export trained models for crypto-quant-system.

Exports models in a portable format that doesn't require
this project's class definitions to load.

Usage:
    python export_for_quant_system.py
    python export_for_quant_system.py --output ../crypto-quant-system/models
"""

import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Export models for quant system")
    parser.add_argument(
        "--output",
        type=str,
        default="../crypto-quant-system/models",
        help="Output directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/BTC.parquet",
        help="Training data path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export Models for crypto-quant-system")
    print("=" * 60)
    print(f"Output: {output_dir}")

    # Load and prepare data
    print("\n[1/4] Loading data...")
    df = load_ohlcv(args.data)
    print(f"  Loaded {len(df)} rows")

    # Extract features
    print("\n[2/4] Extracting features...")
    extractor = FeatureExtractor(include_advanced=False)
    features = extractor.transform(df)
    print(f"  Features: {list(features.columns)}")

    # Generate labels
    print("\n[3/4] Generating labels...")
    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    # Align
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    print(f"  Samples: {len(X)}")
    print(f"  Labels: {dict(y.value_counts())}")

    # Train XGBoost
    print("\n[4/4] Training XGBoost...")
    model = RegimeClassifier(
        model_type="xgboost",
        scale_features=True,
        random_state=42,
    )
    model.fit(X, y, eval_split=0.2, verbose=True)

    # Export in portable format
    print("\n" + "=" * 60)
    print("Exporting...")
    print("=" * 60)

    export_data = {
        # Core model components (no custom classes)
        "model": model.model,  # XGBClassifier
        "scaler": model.scaler,  # StandardScaler
        "label_encoder": model.label_encoder,  # LabelEncoder
        "feature_names": model.feature_names,
        "classes": list(model.label_encoder.classes_),

        # Metadata
        "model_type": "xgboost",
        "n_classes": 2,
        "training_samples": len(X),
        "training_date_range": {
            "start": str(X.index.min().date()),
            "end": str(X.index.max().date()),
        },

        # Performance (from walk-forward validation)
        "performance": {
            "accuracy_mean": 0.8883,
            "accuracy_std": 0.1278,
            "f1_mean": 0.7505,
            "f1_std": 0.3113,
        },
    }

    output_path = output_dir / "regime_classifier_xgb.joblib"
    joblib.dump(export_data, output_path)
    print(f"  Saved: {output_path}")

    # Create usage example
    usage_code = '''"""
Regime Classifier Usage Example
===============================

Load and use the exported XGBoost regime classifier.
"""

import joblib
import pandas as pd
import numpy as np


def load_regime_classifier(path: str = "models/regime_classifier_xgb.joblib"):
    """Load the regime classifier."""
    return joblib.load(path)


def predict_regime(model_data: dict, features: pd.DataFrame) -> pd.Series:
    """Predict market regime.

    Args:
        model_data: Loaded model dictionary
        features: DataFrame with required features

    Returns:
        Series with regime predictions ("BULL_TREND" or "NOT_BULL")
    """
    # Extract components
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]
    feature_names = model_data["feature_names"]

    # Validate features
    missing = set(feature_names) - set(features.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Scale and predict
    X_scaled = scaler.transform(features[feature_names])
    predictions_encoded = model.predict(X_scaled)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    return pd.Series(predictions, index=features.index, name="regime")


def predict_regime_proba(model_data: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Predict regime probabilities.

    Returns:
        DataFrame with columns ["BULL_TREND", "NOT_BULL"]
    """
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    classes = model_data["classes"]

    X_scaled = scaler.transform(features[feature_names])
    probas = model.predict_proba(X_scaled)

    return pd.DataFrame(probas, index=features.index, columns=classes)


# Example usage
if __name__ == "__main__":
    # Load model
    clf = load_regime_classifier("models/regime_classifier_xgb.joblib")

    print("Model Info:")
    print(f"  Type: {clf['model_type']}")
    print(f"  Classes: {clf['classes']}")
    print(f"  Features: {clf['feature_names']}")
    print(f"  Performance: {clf['performance']['accuracy_mean']:.2%} accuracy")

    # To use with real data:
    # features = your_feature_extractor.transform(ohlcv_data)
    # regime = predict_regime(clf, features)
    # proba = predict_regime_proba(clf, features)
'''

    usage_path = output_dir / "regime_classifier_usage.py"
    with open(usage_path, "w") as f:
        f.write(usage_code)
    print(f"  Saved: {usage_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"""
Files created:
  1. {output_path.name}
     - XGBoost model + scaler + label encoder
     - No custom class dependencies

  2. {usage_path.name}
     - Usage example code

Required features for prediction:
  {model.feature_names}

Usage in crypto-quant-system:
  ```python
  import joblib

  clf = joblib.load("models/regime_classifier_xgb.joblib")

  # Predict
  X_scaled = clf["scaler"].transform(features[clf["feature_names"]])
  pred = clf["label_encoder"].inverse_transform(clf["model"].predict(X_scaled))
  ```
""")


if __name__ == "__main__":
    main()
