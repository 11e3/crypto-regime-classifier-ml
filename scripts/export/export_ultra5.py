#!/usr/bin/env python
"""Export XGBoost with ultra-minimal 5 features.

Includes standalone feature calculation function.
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from src.features import FeatureExtractor
from src.labeling import RegimeLabeler
from src.utils.data import load_ohlcv
from src.models import RegimeClassifier
from src.validation.walk_forward import walk_forward_validation


ULTRA5_FEATURES = [
    "return_20d",      # 20일 모멘텀
    "volatility",      # 변동성
    "rsi",             # RSI
    "ma_alignment",    # MA 정렬
    "volume_ratio_20", # 거래량 비율
]


# Standalone feature calculation code (no dependencies)
FEATURE_CALC_CODE = '''"""
Regime Classifier Feature Calculator
=====================================

Calculate 5 features required for regime classification.
No external dependencies except pandas and numpy.
"""

import pandas as pd
import numpy as np


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5 features for regime classification.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
            Index should be datetime

    Returns:
        DataFrame with 5 features:
        - return_20d: 20-day return (momentum)
        - volatility: 20-day rolling volatility
        - rsi: 14-day RSI
        - ma_alignment: MA trend alignment score
        - volume_ratio_20: Volume vs 20-day average
    """
    result = pd.DataFrame(index=df.index)

    # 1. return_20d - 20일 수익률
    result["return_20d"] = df["close"].pct_change(20)

    # 2. volatility - 20일 변동성
    daily_returns = df["close"].pct_change()
    result["volatility"] = daily_returns.rolling(window=20).std()

    # 3. rsi - 14일 RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))

    # 4. ma_alignment - 이평선 정렬 점수
    ma_5 = df["close"].rolling(window=5).mean()
    ma_20 = df["close"].rolling(window=20).mean()
    ma_60 = df["close"].rolling(window=60).mean()

    # 정렬 점수: 5 > 20 > 60이면 +2, 역순이면 -2
    alignment = pd.Series(0, index=df.index, dtype=float)
    alignment += (ma_5 > ma_20).astype(int)
    alignment += (ma_20 > ma_60).astype(int)
    alignment -= (ma_5 < ma_20).astype(int)
    alignment -= (ma_20 < ma_60).astype(int)
    result["ma_alignment"] = alignment

    # 5. volume_ratio_20 - 거래량 비율
    volume_ma_20 = df["volume"].rolling(window=20).mean()
    result["volume_ratio_20"] = df["volume"] / volume_ma_20

    return result


def predict_regime(clf_data: dict, ohlcv_df: pd.DataFrame) -> pd.Series:
    """Predict regime from OHLCV data.

    Args:
        clf_data: Loaded classifier dict from joblib
        ohlcv_df: OHLCV DataFrame

    Returns:
        Series with regime predictions ("BULL_TREND" or "NOT_BULL")
    """
    # Calculate features
    features = calculate_regime_features(ohlcv_df)

    # Drop NaN rows
    features = features.dropna()

    if len(features) == 0:
        raise ValueError("Not enough data to calculate features (need at least 60 rows)")

    # Scale and predict
    X_scaled = clf_data["scaler"].transform(features[clf_data["feature_names"]])
    pred_encoded = clf_data["model"].predict(X_scaled)
    predictions = clf_data["label_encoder"].inverse_transform(pred_encoded)

    return pd.Series(predictions, index=features.index, name="regime")


def predict_regime_proba(clf_data: dict, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Predict regime probabilities.

    Returns:
        DataFrame with columns ["BULL_TREND", "NOT_BULL"]
    """
    features = calculate_regime_features(ohlcv_df)
    features = features.dropna()

    if len(features) == 0:
        raise ValueError("Not enough data")

    X_scaled = clf_data["scaler"].transform(features[clf_data["feature_names"]])
    probas = clf_data["model"].predict_proba(X_scaled)

    return pd.DataFrame(
        probas,
        index=features.index,
        columns=clf_data["classes"]
    )


# Example usage
if __name__ == "__main__":
    import joblib

    # Load model
    clf = joblib.load("models/regime_classifier_xgb_ultra5.joblib")

    print("Model Info:")
    print(f"  Features: {clf['feature_names']}")
    print(f"  Classes: {clf['classes']}")
    print(f"  Performance: {clf['performance']['accuracy_mean']:.2%} accuracy")

    # Example with sample data
    # ohlcv = pd.read_parquet("data/BTC.parquet")
    # regime = predict_regime(clf, ohlcv)
    # print(regime.tail())
'''


def main():
    output_dir = Path("../crypto-quant-system/models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export XGBoost with Ultra-5 Features")
    print("=" * 60)

    # Load and prepare data
    df = load_ohlcv("data/BTC.parquet")
    extractor = FeatureExtractor(include_advanced=False)
    features_full = extractor.transform(df)
    features = features_full[ULTRA5_FEATURES].copy()

    labeler = RegimeLabeler(n_classes=2)
    labels = labeler.label(df)

    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    print(f"Features: {ULTRA5_FEATURES}")
    print(f"Samples: {len(X)}")

    # Walk-forward validation
    print("\nWalk-forward validation...")
    results = walk_forward_validation(
        features=X,
        labels=y,
        model_class=RegimeClassifier,
        model_params={"model_type": "xgboost", "scale_features": True, "random_state": 42},
        train_size=500,
        test_size=60,
        step_size=60,
        verbose=False,
    )
    print(f"Accuracy: {results['accuracy'].mean():.4f} ± {results['accuracy'].std():.4f}")
    print(f"F1:       {results['f1'].mean():.4f} ± {results['f1'].std():.4f}")

    # Train final model
    print("\nTraining final model...")
    model = RegimeClassifier(
        model_type="xgboost",
        scale_features=True,
        random_state=42,
    )
    model.fit(X, y, eval_split=0.2, verbose=False)

    # Export
    export_data = {
        "model": model.model,
        "scaler": model.scaler,
        "label_encoder": model.label_encoder,
        "feature_names": ULTRA5_FEATURES,
        "classes": list(model.label_encoder.classes_),
        "model_type": "xgboost",
        "n_features": 5,
        "performance": {
            "accuracy_mean": float(results['accuracy'].mean()),
            "accuracy_std": float(results['accuracy'].std()),
            "f1_mean": float(results['f1'].mean()),
            "f1_std": float(results['f1'].std()),
        },
    }

    # Save model
    model_path = output_dir / "regime_classifier_xgb_ultra5.joblib"
    joblib.dump(export_data, model_path)
    print(f"\nSaved: {model_path}")

    # Save feature calculator
    calc_path = output_dir / "regime_feature_calculator.py"
    with open(calc_path, "w", encoding="utf-8") as f:
        f.write(FEATURE_CALC_CODE)
    print(f"Saved: {calc_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Files:
  1. regime_classifier_xgb_ultra5.joblib - XGBoost 모델
  2. regime_feature_calculator.py - 피처 계산 함수

사용법 (crypto-quant-system):
```python
import joblib
from regime_feature_calculator import predict_regime

clf = joblib.load("models/regime_classifier_xgb_ultra5.joblib")
regime = predict_regime(clf, ohlcv_df)
# "BULL_TREND" 또는 "NOT_BULL"
```

필요한 OHLCV 컬럼: open, high, low, close, volume
최소 데이터: 60일 (MA60 계산용)
""")


if __name__ == "__main__":
    main()
